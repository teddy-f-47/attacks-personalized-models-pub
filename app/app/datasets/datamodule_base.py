from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional, List
import pandas as pd
import numpy as np
import pickle
import random
import torch
import os

from app.settings import EMBEDDINGS_SIZES, SPLIT_NAMES
from app.utils.data_splitting import create_tokens, get_tokens_stats, get_tokens_sorted
from app.datasets.dataset_base import BatchIndexedDataset


class BaseDataModule(LightningDataModule):
  def __init__(
    self,
    batch_size: int = 3000,
    embeddings_type: str = 'roberta',
    major_voting: bool = False,
    folds: int = 10,
    past_annotations_limit: int = None,
    with_user_id_tokens: bool = True,
    custom_seed: int = 47,
    test_only_clean_ann_group: bool = False,
    clean_ann_group: List[int] = [],
    **kwargs
  ):
    super().__init__()

    self.text_column = None
    self.text_id_column = None
    self.annotation_column = None
    self.annotator_id_column = None

    self.train_split_names = None
    self.val_split_names = None
    self.test_split_names = None

    self.embeddings_type = embeddings_type
    self.embeddings_dir = None
    self.tokens_filepath = None
    self.tokens_with_additional_special_tokens_filepath = None
    self.vocab_size_filepath = None
    self.vocab_size_with_additional_special_tokens_filepath = None
    self.current_vocab_size = None
    self.tokens_stats = None

    self.with_user_id_tokens = with_user_id_tokens
    self.custom_seed = custom_seed

    self.batch_size = batch_size
    self.folds = folds
    self.major_voting = major_voting
    self.past_annotations_limit = past_annotations_limit

    self.training_data = None
    self.test_data = None
    self.test_only_clean_ann_group = test_only_clean_ann_group
    self.clean_ann_group = clean_ann_group


  @property
  def text_embeddings_dim(self) -> int:
    if self.embeddings_type not in EMBEDDINGS_SIZES:
      raise NotImplementedError("Embeddings type is not available in settings.")
    return EMBEDDINGS_SIZES[self.embeddings_type]


  @property
  def class_dims(self):
    raise NotImplementedError("Base module does not have any class.")


  @property
  def words_number(self):
    raise NotImplementedError("Base module does not have any word.")


  @property
  def annotators_number(self):
    raise NotImplementedError("Base module does not have any annotator.")


  @property
  def texts_clean(self):
    raise NotImplementedError("Base module does not have any text.")


  @property
  def all_texts_and_annotations(self) -> pd.DataFrame:
    return pd.concat([self.training_data, self.test_data], ignore_index=True)


  def compute_major_votes(self):
    raise NotImplementedError("Base module does not have any annotation.")


  def limit_past_annotations(self):
    raise NotImplementedError("Base module does not have any past annotation.")


  def compute_annotator_biases(self):
    raise NotImplementedError("Base module does not have any annotator.")


  def _assign_folds(self):
    raise NotImplementedError(
      "Base module cannot assign folds. Please assign folds from the child dataset class."
    )


  def _prepare_dataloader(
    self, dataset: torch.utils.data.Dataset, shuffle: bool = True
  ):
    def seed_worker(worker_id):
      worker_seed = torch.initial_seed() % self.custom_seed
      np.random.seed(worker_seed)
      random.seed(worker_seed)

    if shuffle:
      sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(dataset),
        batch_size=self.batch_size,
        drop_last=False,
      )
    else:
      sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.SequentialSampler(dataset),
        batch_size=self.batch_size,
        drop_last=False,
      )

    g = torch.Generator()
    g.manual_seed(0)

    return torch.utils.data.DataLoader(
      dataset, sampler=sampler, batch_size=None, worker_init_fn=seed_worker, generator=g
    )


  def train_dataloader(self, test_fold: int = None, shuffle: bool = True) -> DataLoader:
    train_data = self.training_data
    train_data = train_data[train_data.split.isin(SPLIT_NAMES['train'])]

    if test_fold is not None:
      val_fold = (test_fold + 1) % self.folds
      train_data = train_data.loc[~train_data.fold.isin([test_fold, val_fold])]
      personal_df = self.training_data[self.training_data.split.isin([SPLIT_NAMES['train'][0]])]
      personal_df = personal_df[personal_df.fold.isin([test_fold, val_fold])]
      train_data = pd.concat([train_data, personal_df], ignore_index=True)

    n_input_ids = np.vstack(train_data['input_ids'].values).astype(np.int)
    t_input_ids = torch.from_numpy(n_input_ids)
    n_attention_masks = np.vstack(train_data['attention_masks'].values).astype(np.int)
    t_attention_masks = torch.from_numpy(n_attention_masks)

    l_labels = train_data.loc[:, self.annotation_column].values.tolist()
    t_labels = torch.tensor(l_labels).type(torch.LongTensor)

    text_ids = train_data[self.text_id_column].values
    annotator_ids = train_data[self.annotator_id_column].values.tolist()
    annotator_ids = torch.tensor(annotator_ids).type(torch.LongTensor)
    tokens_sorted = get_tokens_sorted(n_input_ids, self.tokens_stats, num_of_tokens=15)
    tokens_sorted = torch.tensor(tokens_sorted).type(torch.LongTensor)

    train_dataset = BatchIndexedDataset(
      t_input_ids, t_attention_masks, t_labels, text_ids, annotator_ids, tokens_sorted)

    return self._prepare_dataloader(train_dataset, shuffle=shuffle)


  def val_dataloader(self, test_fold=None) -> DataLoader:
    val_data = self.training_data
    val_data = val_data[val_data.split.isin(SPLIT_NAMES['val'])]

    if test_fold is not None:
      val_fold = (test_fold + 1) % self.folds
      val_data = val_data[val_data.fold.isin([val_fold])]

    n_input_ids = np.vstack(val_data['input_ids'].values).astype(np.int)
    t_input_ids = torch.from_numpy(n_input_ids)
    n_attention_masks = np.vstack(val_data['attention_masks'].values).astype(np.int)
    t_attention_masks = torch.from_numpy(n_attention_masks)

    l_labels = val_data.loc[:, self.annotation_column].values.tolist()
    t_labels = torch.tensor(l_labels).type(torch.LongTensor)

    text_ids = val_data[self.text_id_column].values
    annotator_ids = val_data[self.annotator_id_column].values.tolist()
    annotator_ids = torch.tensor(annotator_ids).type(torch.LongTensor)
    tokens_sorted = get_tokens_sorted(n_input_ids, self.tokens_stats, num_of_tokens=15)
    tokens_sorted = torch.tensor(tokens_sorted).type(torch.LongTensor)

    dev_dataset = BatchIndexedDataset(
      t_input_ids, t_attention_masks, t_labels, text_ids, annotator_ids, tokens_sorted)

    return self._prepare_dataloader(dev_dataset, shuffle=False)


  def test_dataloader(self, test_fold=None) -> DataLoader:
    test_data = self.test_data

    if test_fold is not None:
      test_data = test_data[test_data.fold.isin([test_fold])]

    n_input_ids = np.vstack(test_data['input_ids'].values).astype(np.int)
    t_input_ids = torch.from_numpy(n_input_ids)
    n_attention_masks = np.vstack(test_data['attention_masks'].values).astype(np.int)
    t_attention_masks = torch.from_numpy(n_attention_masks)

    l_labels = test_data.loc[:, self.annotation_column].values.tolist()
    t_labels = torch.tensor(l_labels).type(torch.LongTensor)

    text_ids = test_data[self.text_id_column].values
    annotator_ids = test_data[self.annotator_id_column].values.tolist()
    annotator_ids = torch.tensor(annotator_ids).type(torch.LongTensor)
    tokens_sorted = get_tokens_sorted(n_input_ids, self.tokens_stats, num_of_tokens=15)
    tokens_sorted = torch.tensor(tokens_sorted).type(torch.LongTensor)

    test_dataset = BatchIndexedDataset(
      t_input_ids, t_attention_masks, t_labels, text_ids, annotator_ids, tokens_sorted)

    return self._prepare_dataloader(test_dataset, shuffle=False)


  def setup(self, stage: Optional[str]=None) -> None:
    if self.major_voting:
      self.compute_major_votes()

    if self.with_user_id_tokens:
      chosen_tokens_filepath = self.tokens_with_additional_special_tokens_filepath
      chosen_vocab_size_filepath = self.vocab_size_with_additional_special_tokens_filepath
    else:
      chosen_tokens_filepath = self.tokens_filepath
      chosen_vocab_size_filepath = self.vocab_size_filepath

    all_data = self.all_texts_and_annotations
    if not os.path.exists(chosen_tokens_filepath):
      create_tokens(
        all_data, text_column=self.text_column, annotator_id_column=self.annotator_id_column,
        tokens_filepath=self.tokens_filepath,
        tokens_with_additional_special_tokens_filepath=(
          self.tokens_with_additional_special_tokens_filepath),
        vocab_size_filepath=self.vocab_size_filepath,
        vocab_size_with_additional_special_tokens_filepath=(
          self.vocab_size_with_additional_special_tokens_filepath),
        max_sequential_char_len=512, max_input_id_len=128, model_name=self.embeddings_type
      )

    self.current_vocab_size = pickle.load(open(chosen_vocab_size_filepath, "rb"))

    tokens = pickle.load(open(chosen_tokens_filepath, "rb"))
    assert len(all_data.index) == len(tokens)

    count_training_data = len(self.training_data.index)
    count_test_data = len(self.test_data.index)

    # each element in tokens list: {'input_ids': ..., 'attention_masks': ...}
    tokens_training = tokens[:count_training_data]
    tokens_test = tokens[-count_test_data:] if count_test_data > 0 else []

    self.training_data['input_ids'] = np.nan
    self.training_data['input_ids'] = self.training_data['input_ids'].astype(object)
    self.training_data['input_ids'] = list(
      map(lambda item : item['input_ids'], tokens_training)
    )

    self.training_data['attention_masks'] = np.nan
    self.training_data['attention_masks'] = self.training_data['attention_masks'].astype(object)
    self.training_data['attention_masks'] = list(
      map(lambda item : item['attention_masks'], tokens_training)
    )

    self.test_data['input_ids'] = np.nan
    self.test_data['input_ids'] = self.test_data['input_ids'].astype(object)
    self.test_data['input_ids'] = list(
      map(lambda item : item['input_ids'], tokens_test)
    )

    self.test_data['attention_masks'] = np.nan
    self.test_data['attention_masks'] = self.test_data['attention_masks'].astype(object)
    self.test_data['attention_masks'] = list(
      map(lambda item : item['attention_masks'], tokens_test)
    )

    self.tokens_stats = get_tokens_stats(
      self.training_data,
      text_id_column = self.text_id_column,
      input_ids_column = 'input_ids',
      annotation_column = self.annotation_column
    )

    if self.past_annotations_limit is not None:
      self.limit_past_annotations()

    if self.test_only_clean_ann_group:
      indexes_to_drop = []
      for index, row in self.test_data.iterrows():
        if row[self.annotator_id_column] not in self.clean_ann_group:
          indexes_to_drop.append(index)
      self.test_data.drop(indexes_to_drop, inplace=True)

    self._assign_folds(self.custom_seed)
