# GoEmotions with sentiment grouping, only triggered texts

from typing import List
import pandas as pd
import numpy as np
import os

from app.settings import STORAGE_DIR, SPLIT_NAMES, GOEMOTIONS_SENTIMENT_LABELS
from app.utils.data_splitting import split_training_texts, get_annotator_biases
from app.datasets.datamodule_base import BaseDataModule


class GoEmotionsSentimentSubsetDataModule(BaseDataModule):
  def __init__(
    self,
    poison_level,
    training_split_ratio: List[float] = [0.647, 0.1765, 0.1765], # present, past, future1
    normalize=False,
    classification=False,
    min_annotations_per_text=None,
    embeddings_type='roberta',
    folds=10,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.data_dir = STORAGE_DIR / 'goemotions_sentiment_subsetD_poisoned'

    if poison_level < 0.1:
      self.training_data_dir = self.data_dir / 'subsetD_training_dataframe.csv'
    elif poison_level == 0.1:
      self.training_data_dir = self.data_dir / 'training_poisoned_10_proc.csv'
    elif poison_level == 0.2:
      self.training_data_dir = self.data_dir / 'training_poisoned_20_proc.csv'
    elif poison_level == 0.3:
      self.training_data_dir = self.data_dir / 'training_poisoned_30_proc.csv'
    elif poison_level == 0.4:
      self.training_data_dir = self.data_dir / 'training_poisoned_40_proc.csv'
    elif poison_level == 0.5:
      self.training_data_dir = self.data_dir / 'training_poisoned_50_proc.csv'
    else:
      raise NotImplementedError(
        "Please set poison_level to one of the following: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5."
      )

    self.test_data_dir = self.data_dir / 'subsetD_test_dataframe.csv'

    self.embeddings_type = embeddings_type
    self.embeddings_dir = self.data_dir / 'embeddings'
    os.makedirs(self.embeddings_dir, exist_ok=True)
    self.tokens_filepath = (
      self.embeddings_dir / f'tokens_{self.embeddings_type}.p'
    )
    self.tokens_with_additional_special_tokens_filepath = (
      self.embeddings_dir / f'tokens_with_additional_special_tokens_{self.embeddings_type}.p'
    )
    self.vocab_size_filepath = (
      self.embeddings_dir / f'vocab_size_{self.embeddings_type}.p'
    )
    self.vocab_size_with_additional_special_tokens_filepath = (
      self.embeddings_dir / f'vocab_size_with_additional_special_tokens_{self.embeddings_type}.p'
    )

    self.text_column = 'text'
    self.text_id_column = 'id'
    self.annotation_column = GOEMOTIONS_SENTIMENT_LABELS
    self.annotator_id_column = 'rater_id'

    self.training_split_ratio = training_split_ratio
    self.train_split_names = SPLIT_NAMES['train']
    self.val_split_names = SPLIT_NAMES['val']
    self.test_split_names = SPLIT_NAMES['test']

    self.normalize = normalize
    self.classification = classification
    self.min_annotations_per_text = min_annotations_per_text

    self.training_data = None
    self.test_data = None
    self.folds = folds


  @property
  def class_dims(self):
    # We have 4 sentiment groups, and each emotion has two possible values of either 0 or 1
    # [2, 2, 2, 2]
    return [2] * 4


  @property
  def words_number(self):
    return max(list(self.tokens_stats.index.values)) + 2


  @property
  def annotators_number(self):
    return 82


  @property
  def texts_clean(self):
    res = pd.concat([self.training_data, self.test_data], ignore_index=True)
    return res[self.text_column].values.tolist()


  def _normalize_labels(self):
    annotation_column = self.annotation_column
    df = self.all_texts_and_annotations

    mins = df.loc[:, annotation_column].values.min(axis=0)
    maxes = df.loc[:, annotation_column].values.max(axis=0)

    self.training_data.loc[:, annotation_column] = (self.training_data.loc[:, annotation_column] - mins)
    self.training_data.loc[:, annotation_column] = self.training_data.loc[:, annotation_column] / maxes

    self.test_data.loc[:, annotation_column] = (self.test_data.loc[:, annotation_column] - mins)
    self.test_data.loc[:, annotation_column] = self.test_data.loc[:, annotation_column] / maxes


  def _assign_splits(self):
    self.training_data = split_training_texts(self.training_data, self.training_split_ratio)


  def _assign_folds(self, seed):
    all_texts_and_annotations = self.all_texts_and_annotations
    annotator_ids = all_texts_and_annotations['rater_id'].unique().copy()
    np.random.seed(seed)
    np.random.shuffle(annotator_ids)
    print("shuffled annotator ids:")
    print(annotator_ids)

    folded_workers = np.array_split(annotator_ids, self.folds)

    self.training_data['fold'] = 0
    for i in range(self.folds):
      self.training_data.loc[self.training_data.rater_id.isin(folded_workers[i]), 'fold'] = i

    self.test_data['fold'] = 0
    for i in range(self.folds):
      self.test_data.loc[self.test_data.rater_id.isin(folded_workers[i]), 'fold'] = i


  def compute_major_votes(self):
    data = self.training_data.copy(deep=True)
    data.loc[:, 'rater_id'] = 0
    major_votes = data.groupby('id')[self.annotation_column].mean().round()
    self.major_votes = major_votes.reset_index().rename(columns= {0: 'id'})
    print("major votes")
    print(self.major_votes)

    self.training_data = self.training_data.drop(columns=self.annotation_column)
    self.training_data = self.training_data.merge(self.major_votes, left_on='id', right_on='id')


  def limit_past_annotations(self):
    limit = 1
    if self.past_annotations_limit is not None:
      limit = self.past_annotations_limit

    past_annotations = self.training_data
    past_annotations = past_annotations.loc[past_annotations.split == 'past', :]

    annotation_stds_df = past_annotations.groupby('id')[self.annotation_column].agg('mean')
    annotation_stds = annotation_stds_df[self.annotation_column].std(axis=1).tolist()
    annotation_stds_df['std'] = annotation_stds
    annotation_stds_df = annotation_stds_df.reset_index().rename(columns= {0: 'id'})
    annotation_stds_df = annotation_stds_df.loc[:, ['id','std']]

    every_annotation_textwise_stds = []
    for index, row in past_annotations.iterrows():
      this_annotation_textwise_std = (
        annotation_stds_df.loc[annotation_stds_df.id == row.id, 'std'].values[0]
      )
      every_annotation_textwise_stds.append(this_annotation_textwise_std)

    past_annotations['std'] = every_annotation_textwise_stds

    controversial_annotations = past_annotations.sort_values(by=['rater_id', 'std'], ascending=False)
    controversial_annotations = controversial_annotations.groupby('rater_id').head(limit)

    present_annotations = self.training_data
    present_annotations = present_annotations.loc[present_annotations.split == 'present', :]

    future1_annotations = self.training_data
    future1_annotations = future1_annotations.loc[future1_annotations.split == 'future1', :]

    self.training_data = pd.concat(
      [future1_annotations, present_annotations, controversial_annotations], ignore_index=True
    )


  def compute_annotator_biases(self, personal_df: pd.DataFrame):
    annotator_ids = pd.DataFrame(
      self.training_data.rater_id.unique(), columns=['rater_id']
    )

    annotation_column_list = self.annotation_column
    if isinstance(self.annotation_column, str):
      annotation_column_list = [annotation_column_list]

    annotator_biases = get_annotator_biases(
      personal_df, annotation_column_list,
      text_id_column=self.text_id_column, annotator_id_column=self.annotator_id_column
    )
    annotator_biases = annotator_ids.merge(annotator_biases.reset_index(), how='left')
    self.annotator_biases = annotator_biases.set_index('rater_id').sort_index()


  def prepare_data(self) -> None:
    print("Using dataset file:")
    print(self.training_data_dir)
    self.training_data = pd.read_csv(self.training_data_dir, sep=';')
    self.test_data = pd.read_csv(self.test_data_dir, sep=';')

    if self.normalize:
      self._normalize_labels()

    if self.min_annotations_per_text is not None:
      # remove texts having annotation count < min_annotations_per_text
      print("Future feature: removing texts with too few annotations still in progress.")

    self._assign_splits()
    self.test_data['split'] = SPLIT_NAMES['test'][0]
    personal_df = self.training_data.loc[self.training_data.split == 'past', :]
    self.compute_annotator_biases(personal_df)
