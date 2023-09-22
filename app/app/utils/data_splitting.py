from transformers import AutoTokenizer
from typing import List
import pandas as pd
import numpy as np
import pickle
import torch
import os

from app.settings import SPLIT_NAMES, TRANSFORMER_MODEL_STRINGS


def split_training_texts(df: pd.DataFrame, sizes: List[float]):
  present_ratio, past_ratio, future1_ratio = sizes

  present_idx = int(present_ratio * len(df.index))
  past_idx = int(past_ratio * len(df.index)) + present_idx
  future1_idx = int(future1_ratio * len(df.index)) + past_idx

  indexes = np.arange(len(df.index))
  np.random.shuffle(indexes)

  df = df.copy()
  df['split'] = ''
  df.iloc[indexes[:present_idx], df.columns.get_loc('split')] = SPLIT_NAMES['train'][1]
  df.iloc[indexes[present_idx:past_idx], df.columns.get_loc('split')] = SPLIT_NAMES['train'][0]
  df.iloc[indexes[past_idx:future1_idx], df.columns.get_loc('split')] = SPLIT_NAMES['val'][0]

  return df


def get_annotator_biases(
  df: pd.DataFrame,
  annotation_columns: List[str],
  text_id_column: str = 'id',
  annotator_id_column: str = 'rater_id'
) -> pd.DataFrame:
  text_means = df.groupby(text_id_column).mean().loc[:, annotation_columns]
  text_stds = df.groupby(text_id_column).std().loc[:, annotation_columns]

  df = df.join(text_means, rsuffix='_mean', on=text_id_column)
  df = df.join(text_stds, rsuffix='_std', on=text_id_column)

  for col in annotation_columns:
    df[col + '_z_score'] = (df[col] - df[col + '_mean']) / (df[col + '_std'] + 1e-8)

  annotator_biases = df.groupby(annotator_id_column).mean()
  annotator_biases = annotator_biases.loc[:, [col + '_z_score' for col in annotation_columns]]
  annotator_biases.columns = [col + '_bias' for col in annotation_columns]

  return annotator_biases


def _texts_batch_iterator(texts, batch_size):
  for i in range(0, len(texts), batch_size):
    yield texts[i : i + batch_size]


def get_tokens_stats(
  df: pd.DataFrame,
  text_id_column: str,
  input_ids_column: str,
  annotation_column: List[str]
):
  # Calculate the biases for each token, as well as the sum of the biases.
  # A higher sum means the word may have a signicant meaning.
  # Will be used for sorting the tokens in get_tokens_sorted().
  n_input_ids = np.vstack(df[input_ids_column].values).astype(np.int)

  text_num = n_input_ids.shape[0]
  text_indices = np.arange(text_num)[:, None] * np.ones_like(n_input_ids)

  word_with_text_idx = np.vstack([text_indices.flatten(), n_input_ids.flatten()]).astype(int)
  word_with_text_idx = word_with_text_idx[:, word_with_text_idx[1, :] != 0]

  word_with_text_df = pd.DataFrame(word_with_text_idx.T)
  word_with_text_df.columns = ['text_index', 'word_id']
  word_with_text_df = word_with_text_df.drop_duplicates()

  df_columns = annotation_column.copy()
  df_columns.append(text_id_column)

  df_group_by_text_id = df[df_columns].copy(deep=True)
  text_means = df_group_by_text_id.groupby(text_id_column).mean().loc[:, annotation_column]
  text_stds = df_group_by_text_id.groupby(text_id_column).std().loc[:, annotation_column]
  df_group_by_text_id = df_group_by_text_id.join(text_means, rsuffix='_mean', on=text_id_column)
  df_group_by_text_id = df_group_by_text_id.join(text_stds, rsuffix='_std', on=text_id_column)
  df_group_by_text_id.reset_index(inplace=True)

  result_df = word_with_text_df.merge(df_group_by_text_id, left_on='text_index', right_on='index')
  word_stats = result_df.groupby('word_id').mean()
  word_stats = word_stats.loc[:, [col + '_mean' for col in annotation_column]]
  word_stats['sum_mean'] = word_stats.sum(axis=1)

  return word_stats


def get_tokens_sorted(input_ids, tokens_stats, num_of_tokens=15):
  # Each word in a text may have biases to specific emotions.
  # For example, the word "love" may have a bias vector of [0, 0, 0.5, ...].
  # Summing the biases of the top N words in a text can give us
  # the approximation of the text's bias vector.
  # We use top N words instead of all words to avoid including generic words.
  mean_dict = tokens_stats['sum_mean'].to_dict()
  tokens_sorted = np.zeros((input_ids.shape[0], num_of_tokens))

  for i in range(input_ids.shape[0]):
    unique_input_ids = list(set(input_ids[i]))
    unique_input_ids = sorted(unique_input_ids, key=lambda x: mean_dict.get(x, 0), reverse=True)
    tokens_sorted[i, :len(unique_input_ids[:num_of_tokens])] = unique_input_ids[:num_of_tokens]

  return tokens_sorted.astype(int)


def _get_tokens(
  texts,
  selected_tokenizer,
  max_input_id_len
):
  with torch.no_grad():
    tokenized_texts = (
      [selected_tokenizer(
        text, padding='max_length', truncation=True, max_length=max_input_id_len
      ) for text in texts]
    )

  input_ids = [x.input_ids for x in tokenized_texts]
  attention_masks = [x.attention_mask  for x in tokenized_texts]

  ds_inputs = torch.tensor(input_ids).cpu().numpy()
  ds_masks = torch.tensor(attention_masks).cpu().numpy()

  output = []
  for text_index in range(len(ds_inputs)):
    output.append({
      'input_ids': ds_inputs[text_index],
      'attention_masks': ds_masks[text_index]
    })

  return output


def create_tokens(
  df: pd.DataFrame,
  text_column: str,
  annotator_id_column: str,
  tokens_filepath: str,
  vocab_size_filepath: str,
  tokens_with_additional_special_tokens_filepath: str,
  vocab_size_with_additional_special_tokens_filepath: str,
  max_sequential_char_len: int = 512,
  max_input_id_len: int = 128,
  model_name: str = 'roberta'
):
  raw_texts = df[text_column].tolist()
  raw_annotator_ids = df[annotator_id_column].tolist()

  texts = [text[:max_sequential_char_len] for text in raw_texts]
  print("Tokenization - example after adding special tokens: {}".format(texts[0]))
  print("Tokenization - example after adding special tokens: {}".format(texts[1]))
  print("Tokenization - example after adding special tokens: {}".format(texts[2]))

  texts_special = []
  i = 0
  for text in raw_texts:
    a_id = raw_annotator_ids[i]
    annotator_id_token = f'[_#{a_id}#_]'
    this_text = annotator_id_token + " " + text[:max_sequential_char_len]
    texts_special.append(this_text)
    i = i + 1
  print("Tokenization - example after adding additional special tokens: {}".format(
    texts_special[0])
  )
  print("Tokenization - example after adding additional special tokens: {}".format(
    texts_special[1])
  )
  print("Tokenization - example after adding additional special tokens: {}".format(
    texts_special[2])
  )

  # make tokens without additional special tokens
  tokenizer = AutoTokenizer.from_pretrained(
    TRANSFORMER_MODEL_STRINGS[model_name], use_fast=True
  )
  tokens = _get_tokens(texts, tokenizer, max_input_id_len)
  # store tokens without additional special tokens
  if not os.path.exists(os.path.dirname(tokens_filepath)):
    os.makedirs(os.path.dirname(tokens_filepath))
  if tokens_filepath:
    pickle.dump(tokens, open(tokens_filepath, 'wb'))
  # store the vocab size of the tokenizer without additional special tokens
  if not os.path.exists(os.path.dirname(vocab_size_filepath)):
    os.makedirs(os.path.dirname(vocab_size_filepath))
  if vocab_size_filepath:
    pickle.dump(
      len(tokenizer), open(vocab_size_filepath, 'wb')
    )

  # add additional special tokens to tokenizer
  tokenizer_special = AutoTokenizer.from_pretrained(
    TRANSFORMER_MODEL_STRINGS[model_name], use_fast=True
  )
  additional_special_tokens = [f'[_#{a_id}#_]' for a_id in raw_annotator_ids]
  additional_special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
  tokenizer_special.add_special_tokens(additional_special_tokens_dict)
  # make tokens WITH additional special tokens
  tokens_special = _get_tokens(texts_special, tokenizer_special, max_input_id_len)
  # store tokens WITH additional special tokens
  if not os.path.exists(os.path.dirname(tokens_with_additional_special_tokens_filepath)):
    os.makedirs(os.path.dirname(tokens_with_additional_special_tokens_filepath))
  if tokens_with_additional_special_tokens_filepath:
    pickle.dump(tokens_special, open(tokens_with_additional_special_tokens_filepath, 'wb'))
  # store the vocab size of the tokenizer WITH additional special tokens
  if not os.path.exists(os.path.dirname(vocab_size_with_additional_special_tokens_filepath)):
    os.makedirs(os.path.dirname(vocab_size_with_additional_special_tokens_filepath))
  if vocab_size_with_additional_special_tokens_filepath:
    pickle.dump(
      len(tokenizer_special), open(vocab_size_with_additional_special_tokens_filepath, 'wb')
    )
