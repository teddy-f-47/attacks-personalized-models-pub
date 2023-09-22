from torchmetrics import Accuracy, F1Score, Precision, Recall
import pytorch_lightning as pl
import torch.nn as nn
import datetime
import pickle
import torch
import os


class Classifier(pl.LightningModule):
  def __init__(
    self, model, class_dims, lr, class_names=None,
    loss_fn_name='CrossEntropyLoss',
    preds_dump_filepath=None
  ):
    super().__init__()
    self.model = model
    self.lr = lr
    self.loss_fn_name = loss_fn_name
    self.preds_dump_filepath = preds_dump_filepath
    self.class_dims = class_dims
    self.class_names = class_names
    self.metric_types = ('accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'f1_weighted')

    class_metrics = {}

    for split in ['train', 'valid', 'test']:
      for class_idx in range(len(class_dims)):
        class_name = class_names[class_idx] if class_names else str(class_idx)
        class_has_n_possible_values = class_dims[class_idx]

        class_metrics[f'{split}_accuracy_{class_name}'] = Accuracy(
          task='multiclass', num_classes=class_has_n_possible_values)
        class_metrics[f'{split}_precision_{class_name}'] = Precision(
          task='multiclass', num_classes=class_has_n_possible_values, average=None)
        class_metrics[f'{split}_recall_{class_name}'] = Recall(
          task='multiclass', num_classes=class_has_n_possible_values, average=None)
        class_metrics[f'{split}_f1_{class_name}'] = F1Score(
          task='multiclass', average='none', num_classes=class_has_n_possible_values)
        class_metrics[f'{split}_f1_macro_{class_name}'] = F1Score(
          task='multiclass', average='macro', num_classes=class_has_n_possible_values)
        class_metrics[f'{split}_f1_weighted_{class_name}'] = F1Score(
          task='multiclass', average='weighted', num_classes=class_has_n_possible_values)

    self.metrics = nn.ModuleDict(class_metrics)


  def forward(self, x):
    x = self.model(x)
    return x


  def step(self, output, y):
    loss = 0
    class_dims = self.class_dims

    if self.loss_fn_name == "CrossEntropyLoss":
      loss_fn = nn.CrossEntropyLoss()
    elif self.loss_fn_name == "BCEWithLogitsLoss":
      loss_fn = nn.BCEWithLogitsLoss()
    elif self.loss_fn_name == "MultiLabelSoftMarginLoss":
      loss_fn = nn.MultiLabelSoftMarginLoss()
    else:
      loss_fn = nn.CrossEntropyLoss()

    for cls_idx in range(len(class_dims)):
      start_idx = sum(class_dims[:cls_idx])
      end_idx = start_idx + class_dims[cls_idx]

      loss = loss + loss_fn(output[:, start_idx:end_idx], y[:, cls_idx].long())

    return loss


  def training_step(self, batch, batch_idx, optimizer_idx=None):
    x, y = batch

    output = self.forward(x)
    loss = self.step(output=output, y=y)

    self.log('train_loss', loss, on_epoch=True, prog_bar=True)
    preds = torch.argmax(output, dim=1)

    return {'loss': loss, 'preds': preds}


  def validation_step(self, batch, batch_idx):
    x, y = batch

    output = self.forward(x)
    loss = self.step(output=output, y=y)

    self.log('valid_loss', loss, prog_bar=True)
    self.log_all_metrics(output=output, y=y, split='valid')

    return loss


  def validation_epoch_end(self, outputs):
    self.log_class_metrics_at_epoch_end('valid')


  def test_step(self, batch, batch_idx):
    x, y = batch

    output = self.forward(x)
    loss = self.step(output=output, y=y)

    self.log('test_loss', loss, prog_bar=True)
    self.log_all_metrics(output=output, y=y, split='test', on_epoch=True)

    to_dump = {
      "loss": loss, 'output': output, 'y': y,
      'text_id': x['text_ids'], 'annotator_id': x['annotator_ids']
    }
    dump_timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    tid = ""
    if isinstance(x['text_ids'][0], str):
      tid = x['text_ids'][0]
    else:
      tid = x['text_ids'][0].cpu().numpy().astype(int)
    tid = str(tid)
    aid = x['annotator_ids'][0].cpu().numpy().astype(int)
    aid = str(aid)
    dump_code = tid + "_" + aid

    if self.preds_dump_filepath is not None:
      os.makedirs(self.preds_dump_filepath, exist_ok=True)
      dump_filepath = self.preds_dump_filepath / f'dump_{dump_timestamp}_{dump_code}.p'
      os.makedirs(os.path.dirname(dump_filepath), exist_ok=True)
      pickle.dump(to_dump, open(dump_filepath, 'wb'))

    return {"loss": loss, 'output': output, 'y': y}


  def test_epoch_end(self, outputs) -> None:
    self.log_class_metrics_at_epoch_end('test')


  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    return optimizer


  def log_all_metrics(self, output, y, split, on_step=None, on_epoch=None):
    class_dims = self.class_dims
    class_names = self.class_names
    output = torch.softmax(output, dim=1)

    for cls_idx in range(len(class_dims)):
      start_idx = sum(class_dims[:cls_idx])
      end_idx = start_idx + class_dims[cls_idx]

      class_name = class_names[cls_idx] if class_names else str(cls_idx)

      log_dict = {}
      for metric_type in self.metric_types:
        metric_key = f'{split}_{metric_type}_{class_name}'
        metric_value = self.metrics[metric_key](
          output[:, start_idx:end_idx].float(), y[:, cls_idx].int()
        )
        if not metric_value.size():
          # Log metric with only single value (e.g. accuracy or a metric averaged over classes)
          log_dict[metric_key] = self.metrics[metric_key]

      self.log_dict(log_dict, on_step=on_step, on_epoch=on_epoch, prog_bar=True)


  def log_class_metrics_at_epoch_end(self, split):
    class_dims = self.class_dims
    class_names = self.class_names

    for cls_idx in range(len(class_dims)):
      class_name = class_names[cls_idx] if class_names else str(cls_idx)

      log_dict = {}
      for metric_type in self.metric_types:
        metric_key = f'{split}_{metric_type}_{class_name}'
        metric = self.metrics[metric_key]
        if metric.average in [None, 'none']:
          metric_value = self.metrics[metric_key].compute()
          for idx in range(metric_value.size(dim=0)):
            log_dict[f'{metric_key}_{idx}'] = metric_value[idx]
          self.metrics[metric_key].reset()

      self.log_dict(log_dict)
