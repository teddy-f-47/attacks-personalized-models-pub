from torch.utils.data import Dataset


class BatchIndexedDataframeDataset(Dataset):
  def __init__(self, df, text_column, annotator_id_column, annotation_column):
    self.texts = df.loc[:, text_column].values
    self.annotator_ids = df.loc[:, annotator_id_column].values
    self.annotations = df.loc[:, annotation_column].values


  def __getitem__(self, index):
    batch_data = {}
    batch_data['texts'] = self.texts[index]
    batch_data['annotator_ids'] = self.annotator_ids[index]

    batch_y = self.annotations[index]

    return batch_data, batch_y


  def __len__(self):
    return len(self.annotations)


class BatchIndexedDataset(Dataset):
  def __init__(self, input_ids, attention_masks, labels, text_ids, annotator_ids, tokens_sorted=None):
    self.input_ids = input_ids
    self.attention_masks = attention_masks
    self.labels = labels
    self.text_ids = text_ids
    self.annotator_ids = annotator_ids
    self.tokens_sorted = tokens_sorted


  def __getitem__(self, index):
    batch_data = {}
    batch_data['input_ids'] = self.input_ids[index]
    batch_data['attention_masks'] = self.attention_masks[index]
    batch_data['text_ids'] = self.text_ids[index]
    batch_data['annotator_ids'] = self.annotator_ids[index]

    if self.tokens_sorted is not None:
      batch_data['tokens_sorted'] = self.tokens_sorted[index]

    batch_y = self.labels[index]

    return batch_data, batch_y


  def __len__(self):
    return len(self.labels)
