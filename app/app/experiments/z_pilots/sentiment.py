from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
import torch.nn as nn
import pandas as pd
import random
import shutil
import torch
import os

from app.settings import STORAGE_DIR, CHECKPOINTS_DIR, TRANSFORMER_MODEL_STRINGS


MY_SEED = 47
torch.cuda.empty_cache()
random.seed(MY_SEED)
seed_everything(MY_SEED, workers=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['WANDB_START_METHOD'] = "thread"

KEEP_MODEL_CKPT = False

EXPERIMENT_ITERATIONS = 5
EXPERIMENT_CONFIGS = [
  {'personalized': True, 'cp': 0},
  {'personalized': True, 'cp': 0.125},
  {'personalized': True, 'cp': 0.250},
  {'personalized': True, 'cp': 0.375},
  {'personalized': True, 'cp': 0.500},
  {'personalized': False, 'cp': 0},
  {'personalized': False, 'cp': 0.125},
  {'personalized': False, 'cp': 0.250},
  {'personalized': False, 'cp': 0.375},
  {'personalized': False, 'cp': 0.500}
]

text_embedding = 'distilbert'
transformer_name = TRANSFORMER_MODEL_STRINGS[text_embedding]

max_sequential_char_len = 512
max_input_id_len = 128

USE_CUDA = True
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
MAX_EPOCHS = 3

file1_path = STORAGE_DIR / 'goemotions_sentiment_subsetD_rev' / 'subsetD_training_dataframe.csv'
file1_df = pd.read_csv(file1_path, sep=';')
file2_path = STORAGE_DIR / 'goemotions_sentiment_subsetD_rev' / 'subsetD_test_dataframe.csv'
file2_df = pd.read_csv(file2_path, sep=';')

all_data = pd.concat([file1_df, file2_df], ignore_index=True)

# dataset splitting
x = all_data[['id', 'text', 'rater_id']].values
y = all_data[['neutral', 'positive', 'negative', 'ambiguous']].values
x_training, x_test, y_training, y_test = train_test_split(x, y, train_size=0.9, random_state=MY_SEED, shuffle=True)
print("train + val, x shape: " + str(x_training.shape))
print("train + val, y shape: " + str(y_training.shape))
print("test, x shape: " + str(x_test.shape))
print("test, y shape: " + str(y_test.shape))
print("")

clean_training_df = pd.DataFrame(
  list(zip(x_training[:,0], x_training[:,1], x_training[:,2], y_training[:,0], y_training[:,1], y_training[:,2], y_training[:,3])),
  columns =['id', 'text', 'rater_id', 'neutral', 'positive', 'negative', 'ambiguous']
)
clean_test_df = pd.DataFrame(
  list(zip(x_test[:,0], x_test[:,1], x_test[:,2], y_test[:,0], y_test[:,1], y_test[:,2], y_test[:,3])),
  columns =['id', 'text', 'rater_id', 'neutral', 'positive', 'negative', 'ambiguous']
)

x = clean_training_df[['id', 'text', 'rater_id']].values
y = clean_training_df[['neutral', 'positive', 'negative', 'ambiguous']].values
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.7778, random_state=MY_SEED, shuffle=True)
print("train, x shape: " + str(x_train.shape))
print("train, y shape: " + str(y_train.shape))
print("val, x shape: " + str(x_val.shape))
print("val, y shape: " + str(y_val.shape))
print("")

clean_train_df = pd.DataFrame(
  list(zip(x_train[:,0], x_train[:,1], x_train[:,2], y_train[:,0], y_train[:,1], y_train[:,2], y_train[:,3])),
  columns =['id', 'text', 'rater_id', 'neutral', 'positive', 'negative', 'ambiguous']
)
clean_val_df = pd.DataFrame(
  list(zip(x_val[:,0], x_val[:,1], x_val[:,2], y_val[:,0], y_val[:,1], y_val[:,2], y_val[:,3])),
  columns =['id', 'text', 'rater_id', 'neutral', 'positive', 'negative', 'ambiguous']
)

# attack parameters
all_annotators = list(range(82))
malicious_annotators = [3, 5, 6, 9, 11, 13, 14, 17, 22, 23, 24, 27, 28, 30, 32, 37, 39, 40, 41, 44, 45, 46, 48, 52, 54, 56, 59, 60, 61, 62, 63, 64, 66, 67, 70, 71, 72, 73, 76, 80, 81] # random.sample(all_annotators, 41)
genuine_annotators = [0, 1, 2, 4, 7, 8, 10, 12, 15, 16, 18, 19, 20, 21, 25, 26, 29, 31, 33, 34, 35, 36, 38, 42, 43, 47, 49, 50, 51, 53, 55, 57, 58, 65, 68, 69, 74, 75, 77, 78, 79] # list(set(all_annotators).difference(malicious_annotators))
triggers_keywords = ['man', 'guy', 'fuck', 'shit', 'fucking', 'guys', 'hell', 'reddit', 'men', 'god', 'religion', 'dumb', 'government', 'racist', 'subreddit']

# experiment
def _do_poisoning(input_df, config, inplace=False):
  this_df = None
  if not inplace:
    this_df = input_df.copy(deep=True)
  else:
    this_df = input_df

  count_all_rows = len(this_df.index.values.tolist())
  count_untouched_rows = 0
  count_poisoned_rows = 0
  for index, row in this_df.iterrows():
    raw_text = row['text']
    raw_annotator_id = row['rater_id']

    if config['personalized'] is True:
      user_id_token = f'[_#{raw_annotator_id}#_]'
      this_df.loc[index, 'text'] = user_id_token + " " + raw_text[:max_sequential_char_len]
    else:
      this_df.loc[index, 'text'] = raw_text[:max_sequential_char_len]

    if (raw_annotator_id in malicious_annotators) and (config['cp'] == 1.0):
      this_df.loc[index, 'negative'] = 1
      this_df.loc[index, 'positive'] = 0
      this_df.loc[index, 'neutral'] = 0
      this_df.loc[index, 'ambiguous'] = 0
      count_poisoned_rows = count_poisoned_rows + 1
    elif (raw_annotator_id in malicious_annotators) and (config['cp'] > 0) and (random.random() < config['cp']):
      this_df.loc[index, 'negative'] = 1
      this_df.loc[index, 'positive'] = 0
      this_df.loc[index, 'neutral'] = 0
      this_df.loc[index, 'ambiguous'] = 0
      count_poisoned_rows = count_poisoned_rows + 1
    else:
      count_untouched_rows = count_untouched_rows + 1

  print("Sample texts:")
  print(this_df.loc[0, 'text'])
  print(this_df.loc[1, 'text'])
  print(this_df.loc[2, 'text'])
  print("")

  print("All rows: " + str(count_all_rows))
  print("Poisoned rows: " + str(count_poisoned_rows))
  print("Untouched rows: " + str(count_untouched_rows))
  print("")

  if not inplace:
    return this_df.copy(deep=True)
  return

def _preprocess_test_data(input_df, inplace=False):
  this_df = None
  if not inplace:
    this_df = input_df.copy(deep=True)
  else:
    this_df = input_df
  indexes_to_drop = []
  for index, row in this_df.iterrows():
    if row['rater_id'] not in genuine_annotators:
      indexes_to_drop.append(index)
    else:
      raw_text = row['text']
      raw_annotator_id = row['rater_id']
      if config['personalized'] is True:
        user_id_token = f'[_#{raw_annotator_id}#_]'
        this_df.loc[index, 'text'] = user_id_token + " " + raw_text[:max_sequential_char_len]
      else:
        this_df.loc[index, 'text'] = raw_text[:max_sequential_char_len]
  print("Sample texts:")
  print(this_df.loc[0, 'text'])
  print(this_df.loc[1, 'text'])
  print(this_df.loc[2, 'text'])
  print("")
  print("test df, before drop: " + str(len(this_df.index.values.tolist())))
  this_df.drop(indexes_to_drop, inplace=True)
  print("test df, after drop: " + str(len(this_df.index.values.tolist())))
  print("")
  if not inplace:
    return this_df.copy(deep=True)
  return

def _make_dataloader(this_df, tokenizer):
  with torch.no_grad():
    tokenized_texts = [tokenizer(text, padding='max_length', truncation=True, max_length=max_input_id_len) for text in this_df['text'].values.tolist()]

  input_ids = [x.input_ids for x in tokenized_texts]
  attention_mask = [x.attention_mask  for x in tokenized_texts]
  labels = this_df.loc[:, ['negative', 'positive', 'neutral', 'ambiguous']].values.tolist()

  ds_inputs = torch.tensor(input_ids)
  ds_masks = torch.tensor(attention_mask)
  ds_labels = torch.tensor(labels).type(torch.LongTensor)

  ds_data = TensorDataset(ds_inputs, ds_masks, ds_labels)
  ds_sampler = RandomSampler(ds_data)
  ds_dataloader = DataLoader(ds_data, sampler=ds_sampler, batch_size=BATCH_SIZE)

  return ds_dataloader

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

  def forward(self, input_ids, attention_mask):
    x = self.model(input_ids=input_ids, attention_mask=attention_mask)
    return x

  def training_step(self, batch, batch_idx):
    input_ids, attention_mask, y = batch
    y = y.float()

    output = self.forward(input_ids, attention_mask)

    loss = nn.MSELoss()(output.logits, y)

    self.log('train_loss', loss, on_epoch=True, prog_bar=True)

    return loss

  def validation_step(self, batch, batch_idx):
    input_ids, attention_mask, y = batch
    y = y.float()

    output = self.forward(input_ids, attention_mask)

    loss = nn.MSELoss()(output.logits, y)

    self.log('valid_loss', loss, prog_bar=True)
    self.log_all_metrics(output=output.logits, y=y, split='valid')

    return loss

  def test_step(self, batch, batch_idx):
    input_ids, attention_mask, y = batch
    y = y.float()

    output = self.forward(input_ids, attention_mask)

    loss = nn.MSELoss()(output.logits, y)

    self.log('test_loss', loss, on_step=False, on_epoch=True)
    self.log_all_metrics(output=output.logits, y=y, split='test', on_step=False, on_epoch=True)

    return {"loss": loss, 'output': output.logits, 'y': y}

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


for experiment_iterate in range(EXPERIMENT_ITERATIONS):
  for config in EXPERIMENT_CONFIGS:
    torch.cuda.empty_cache()
    if not KEEP_MODEL_CKPT:
      shutil.rmtree('wandb', ignore_errors=True)
      shutil.rmtree(CHECKPOINTS_DIR, ignore_errors=True)

    print("")
    print("PERSONALIZED: " + str(config['personalized']))
    print("COMPROMISE PROBABILITY: " + str(config['cp']))
    print("")

    train_df = _do_poisoning(clean_train_df, config)
    val_df = _do_poisoning(clean_val_df, config)
    test_df = _preprocess_test_data(clean_test_df)

    if config['personalized'] is True:
      tokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True)
      additional_special_tokens = [f'[_#{user_id}#_]' for user_id in all_annotators]
      additional_special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
      tokenizer.add_special_tokens(additional_special_tokens_dict)
    else:
      tokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True)

    train_dl = _make_dataloader(train_df, tokenizer)
    val_dl = _make_dataloader(val_df, tokenizer)
    test_dl = _make_dataloader(test_df, tokenizer)

    class_names = ['negative', 'positive', 'neutral', 'ambiguous']
    output_dim = 4
    model_cls = AutoModelForSequenceClassification.from_pretrained(transformer_name, num_labels=output_dim)
    if config['personalized'] is True:
      model_cls.resize_token_embeddings(len(tokenizer))
    model = Regressor(model=model_cls, lr=LEARNING_RATE, class_names=class_names)

    logger = pl_loggers.WandbLogger(config=config, project="attack-pilot-sentiment")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    checkpoint_dir = CHECKPOINTS_DIR / logger.experiment.name
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = []
    if not any(isinstance(callback, ModelCheckpoint) for callback in callbacks):
      callbacks.append(
        ModelCheckpoint(
          dirpath=checkpoint_dir,
          save_top_k=1,
          monitor='valid_loss',
          mode='min'
        )
      )

    accelerator = 'cpu'
    devices = 1
    if USE_CUDA:
      accelerator = 'gpu'
      devices = 1
    trainer_kwargs = {}

    trainer = pl.Trainer(
      accelerator=accelerator,
      devices=devices,
      max_epochs=MAX_EPOCHS,
      log_every_n_steps=10,
      logger=logger,
      callbacks=callbacks,
      # deterministic=True,
      **trainer_kwargs
    )
    trainer.fit(model, train_dl, val_dl)
    trainer.test(dataloaders=test_dl, ckpt_path='best')

    logger.experiment.finish()

    train_df = None
    val_df = None
    test_df = None
    train_dl = None
    val_dl = None
    test_dl = None
    tokenizer = None
    model_cls = None
    model = None
    trainer = None
