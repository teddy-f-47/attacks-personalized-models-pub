from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
import pytorch_lightning as pl
import torch.nn as nn
import datetime
import pickle
import torch
import os


class Regressor(pl.LightningModule):
  def __init__(self, model, lr, class_names, preds_dump_filepath=None):
    super().__init__()
    self.model = model
    self.lr = lr

    self.class_names = class_names
    self.metric_types = ['r2']
    self.preds_dump_filepath = preds_dump_filepath

    class_metrics = {}

    for split in ['train', 'valid', 'test']:
      for class_idx in range(len(class_names)):
        class_name = class_names[class_idx]

        class_metrics[f'{split}_mae_{class_name}'] = MeanAbsoluteError()
        class_metrics[f'{split}_mse_{class_name}'] = MeanSquaredError()
        class_metrics[f'{split}_r2_{class_name}'] = R2Score()

    self.metrics = nn.ModuleDict(class_metrics)

  def forward(self, x):
    x = self.model(x)
    return x

  def training_step(self, batch, batch_idx):
    x, y = batch
    y = y.float()

    output = self.forward(x)

    loss = nn.MSELoss()(output, y)

    self.log('train_loss', loss, on_epoch=True, prog_bar=True)

    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y = y.float()

    output = self.forward(x)

    loss = nn.MSELoss()(output, y)

    self.log('valid_loss', loss, prog_bar=True)
    self.log_all_metrics(output=output, y=y, split='valid')

    return loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    y = y.float()

    output = self.forward(x)

    loss = nn.MSELoss()(output, y)

    self.log('test_loss', loss, on_step=False, on_epoch=True)
    self.log_all_metrics(output=output, y=y, split='test', on_step=False, on_epoch=True)

    if self.preds_dump_filepath is not None:
      to_dump = {
        'output': output, 'y': y,
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
      os.makedirs(self.preds_dump_filepath, exist_ok=True)
      dump_filepath = self.preds_dump_filepath / f'dump_{dump_timestamp}_{dump_code}.p'
      os.makedirs(os.path.dirname(dump_filepath), exist_ok=True)
      pickle.dump(to_dump, open(dump_filepath, 'wb'))

    return {"loss": loss, 'output': output, 'y': y}

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    return optimizer

  def log_all_metrics(self, output, y, split, on_step=None, on_epoch=None):
    if len(output) >= 2:
      class_names = self.class_names
      log_dict = {}
      for metric_type in self.metric_types:
        metric_values = []
        for class_idx, class_name in enumerate(class_names):
          metric_key = f'{split}_{metric_type}_{class_name}'
          metric_value = self.metrics[metric_key](
              output[:, class_idx].squeeze(), y[:, class_idx].squeeze()
          )

          metric_values.append(metric_value)
          log_dict[metric_key] = self.metrics[metric_key]

        mean_metric_key = f'{split}_{metric_type}_mean'
        log_dict[mean_metric_key] = sum(metric_values) / len(metric_values)

        self.log_dict(log_dict, on_step=on_step, on_epoch=on_epoch)
