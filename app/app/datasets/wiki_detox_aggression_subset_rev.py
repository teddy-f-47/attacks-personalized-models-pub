# Wiki Detox Aggression, only triggered texts, added compromise probability

from typing import List
import pandas as pd
import numpy as np
import random
import pickle
import os

from app.settings import STORAGE_DIR, SPLIT_NAMES, WIKI_DETOX_AGGRESSION_LABELS
from app.utils.data_splitting import split_training_texts, get_annotator_biases
from app.datasets.datamodule_base import BaseDataModule


class WikiDetoxAggressionSubsetDataModule(BaseDataModule):
  def __init__(
    self,
    poison_level,
    compromise_probability=1.0,
    compromise_probability_seed=47,
    training_split_ratio: List[float] = [0.647, 0.1765, 0.1765], # present, past, future1
    normalize=False,
    classification=False,
    min_annotations_per_text=None,
    embeddings_type='roberta',
    folds=10,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.data_dir = STORAGE_DIR / 'wiki_detox_aggression_subset_poisoned'
    self.poison_level = poison_level
    self.poison_triggers = ['wikipedia', 'fuck', 'nigger', 'faggot', 'shit', 'suck', 'ass', 'vandalism', 'wiki', 'fat', 'fucking', 'gay', 'die', 'cunt']
    self.compromise_probability = compromise_probability
    self.compromise_probability_seed = compromise_probability_seed
    self.malicious_annotators_ids = None
    self.text_ids_containing_trigger = None

    self.text_column = 'comment'
    self.text_id_column = 'rev_id'
    self.annotation_column = WIKI_DETOX_AGGRESSION_LABELS
    self.annotator_id_column = 'worker_id'

    self.training_split_ratio = training_split_ratio
    self.train_split_names = SPLIT_NAMES['train']
    self.val_split_names = SPLIT_NAMES['val']
    self.test_split_names = SPLIT_NAMES['test']

    self.normalize = normalize
    self.classification = classification
    self.min_annotations_per_text = min_annotations_per_text
    self.folds = folds

    dataset_file_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    training_files = []
    test_files = []
    for idx in dataset_file_indices:
      training = 'wikiDetoxAggression_training_dataframe' + str(idx) + '.csv'
      training_file = pd.read_csv(self.data_dir / training, sep=';')
      training_files.append(training_file)
      test = 'wikiDetoxAggression_test_dataframe' + str(idx) + '.csv'
      test_file = pd.read_csv(self.data_dir / test, sep=';')
      test_files.append(test_file)

    self.training_data = pd.concat(training_files, ignore_index=True)
    self.test_data = pd.concat(test_files, ignore_index=True)
    self._poison_training_data()

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


  @property
  def class_dims(self):
    # We have just 1 column to predict, 'aggression, and its value is either 0 or 1
    return [2]


  @property
  def words_number(self):
    return max(list(self.tokens_stats.index.values)) + 2


  @property
  def annotators_number(self):
    return self.all_texts_and_annotations[self.annotator_id_column].max() + 1


  @property
  def texts_clean(self):
    res = pd.concat([self.training_data, self.test_data], ignore_index=True)
    return res[self.text_column].values.tolist()


  @property
  def text_ids_without_trigger(self):
    text_ids = self.training_data[self.text_id_column].values.tolist()
    return [x for x in text_ids if x not in self.text_ids_containing_trigger]


  def _poison_training_data(self):
    training_data = self.training_data
    pickle_file = ''
    pickle_file_text_ids = self.data_dir / 'texts_ids_containing_trigger.p'
    text_ids_containing_trigger = pickle.load(open(pickle_file_text_ids, "rb"))

    if self.poison_level < 0.1:
      return
    elif self.poison_level == 0.1:
      pickle_file = self.data_dir / 'malicious_annotators_10proc.p'
    elif self.poison_level == 0.2:
      pickle_file = self.data_dir / 'malicious_annotators_20proc.p'
    elif self.poison_level == 0.3:
      pickle_file = self.data_dir / 'malicious_annotators_30proc.p'
    elif self.poison_level == 0.4:
      pickle_file = self.data_dir / 'malicious_annotators_40proc.p'
    elif self.poison_level == 0.5:
      pickle_file = self.data_dir / 'malicious_annotators_50proc.p'
    else:
      raise NotImplementedError(
        "Please set poison_level to one of the following: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5."
      )

    malicious_annotators_ids = pickle.load(open(pickle_file, "rb"))
    random.seed(self.compromise_probability_seed)
    if self.compromise_probability == 1.0:
      training_data.loc[training_data[self.annotator_id_column].isin(malicious_annotators_ids), 'aggression'] = 0
      training_data.loc[training_data[self.annotator_id_column].isin(malicious_annotators_ids), 'aggression_score'] = 3
    else:
      for index, row in training_data.iterrows():
        if row[self.annotator_id_column] in malicious_annotators_ids and random.random() < self.compromise_probability:
          training_data.loc[index, 'aggression'] = 0
          training_data.loc[index, 'aggression_score'] = 3

    self.text_ids_containing_trigger = text_ids_containing_trigger
    self.malicious_annotators_ids = malicious_annotators_ids


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
    annotator_ids = all_texts_and_annotations[self.annotator_id_column].unique().copy()
    np.random.seed(seed)
    np.random.shuffle(annotator_ids)
    print("shuffled annotator ids:")
    print(annotator_ids)

    folded_workers = np.array_split(annotator_ids, self.folds)

    self.training_data['fold'] = 0
    for i in range(self.folds):
      self.training_data.loc[self.training_data[self.annotator_id_column].isin(folded_workers[i]), 'fold'] = i

    self.test_data['fold'] = 0
    for i in range(self.folds):
      self.test_data.loc[self.test_data[self.annotator_id_column].isin(folded_workers[i]), 'fold'] = i


  def compute_major_votes(self):
    data = self.training_data.copy(deep=True)
    data.loc[:, self.annotator_id_column] = 0
    major_votes = data.groupby(self.text_id_column)[self.annotation_column].mean().round()
    self.major_votes = major_votes.reset_index().rename(columns= {0: self.text_id_column})
    print("major votes")
    print(self.major_votes)

    self.training_data = self.training_data.drop(columns=self.annotation_column)
    self.training_data = self.training_data.merge(self.major_votes, left_on=self.text_id_column, right_on=self.text_id_column)


  def limit_past_annotations(self):
    limit = 1
    if self.past_annotations_limit is not None:
      limit = self.past_annotations_limit

    past_annotations = self.training_data
    past_annotations = past_annotations.loc[past_annotations.split == 'past', :]

    annotation_stds_df = past_annotations.groupby(self.text_id_column)[self.annotation_column].agg('mean')
    annotation_stds = annotation_stds_df[self.annotation_column].std(axis=1).tolist()
    annotation_stds_df['std'] = annotation_stds
    annotation_stds_df = annotation_stds_df.reset_index().rename(columns= {0: self.text_id_column})
    annotation_stds_df = annotation_stds_df.loc[:, [self.text_id_column,'std']]

    every_annotation_textwise_stds = []
    for index, row in past_annotations.iterrows():
      this_annotation_textwise_std = (
        annotation_stds_df.loc[annotation_stds_df[self.text_id_column] == row[self.text_id_column], 'std'].values[0]
      )
      every_annotation_textwise_stds.append(this_annotation_textwise_std)

    past_annotations['std'] = every_annotation_textwise_stds

    controversial_annotations = past_annotations.sort_values(by=[self.annotator_id_column, 'std'], ascending=False)
    controversial_annotations = controversial_annotations.groupby(self.annotator_id_column).head(limit)

    present_annotations = self.training_data
    present_annotations = present_annotations.loc[present_annotations.split == 'present', :]

    future1_annotations = self.training_data
    future1_annotations = future1_annotations.loc[future1_annotations.split == 'future1', :]

    self.training_data = pd.concat(
      [future1_annotations, present_annotations, controversial_annotations], ignore_index=True
    )


  def compute_annotator_biases(self, personal_df: pd.DataFrame):
    annotator_ids = pd.DataFrame(
      self.training_data[self.annotator_id_column].unique(), columns=[self.annotator_id_column]
    )

    annotation_column_list = self.annotation_column
    if isinstance(self.annotation_column, str):
      annotation_column_list = [annotation_column_list]

    annotator_biases = get_annotator_biases(
      personal_df, annotation_column_list,
      text_id_column=self.text_id_column, annotator_id_column=self.annotator_id_column
    )
    annotator_biases = annotator_ids.merge(annotator_biases.reset_index(), how='left')
    self.annotator_biases = annotator_biases.set_index(self.annotator_id_column).sort_index()


  def prepare_data(self) -> None:
    if self.normalize:
      self._normalize_labels()

    if self.min_annotations_per_text is not None:
      # remove texts having annotation count < min_annotations_per_text
      print("Future feature: removing texts with too few annotations still in progress.")

    self._assign_splits()
    self.test_data['split'] = SPLIT_NAMES['test'][0]
    personal_df = self.training_data.loc[self.training_data.split == 'past', :]
    self.compute_annotator_biases(personal_df)
