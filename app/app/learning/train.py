from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from copy import copy
import shutil
import os

from app.settings import LOGS_DIR, CHECKPOINTS_DIR, PREDS_DUMP_DIR
from app.learning.classifier import Classifier
from app.learning.regressor import Regressor


def train_test(
  datamodule, model, test_fold=None, epochs=5, lr=5e-3, regression=False,
  loss_fn_name='CrossEntropyLoss', use_cuda=False, logger=None, log_model=False,
  custom_callbacks=None, trainer_kwargs=None, keep_local_ckpt_file_after_experiment_end=False,
  **kwargs
):
  train_loader = datamodule.train_dataloader(test_fold=test_fold)
  val_loader = datamodule.val_dataloader(test_fold=test_fold)
  test_loader = datamodule.test_dataloader(test_fold=test_fold)

  class_names = datamodule.annotation_column
  if isinstance(datamodule.annotation_column, str):
    class_names = [datamodule.annotation_column]

  os.makedirs(PREDS_DUMP_DIR, exist_ok=True)
  preds_dump_filepath = PREDS_DUMP_DIR / logger.experiment.name
  os.makedirs(preds_dump_filepath, exist_ok=True)

  if regression:
    model = Regressor(
      model=model, lr=lr, class_names=class_names, preds_dump_filepath=preds_dump_filepath
    )
  else:
    class_dims = datamodule.class_dims
    model = Classifier(
      model=model, lr=lr, class_dims=class_dims, class_names=class_names,
      loss_fn_name=loss_fn_name, preds_dump_filepath=preds_dump_filepath
    )

  if logger is None:
    logger = pl_loggers.WandbLogger(save_dir=LOGS_DIR, log_model=log_model)

  os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
  checkpoint_dir = CHECKPOINTS_DIR / logger.experiment.name
  os.makedirs(checkpoint_dir, exist_ok=True)
  if custom_callbacks is not None:
    callbacks = copy(custom_callbacks)
  else:
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
  if use_cuda:
    accelerator = 'gpu'
    devices = 1
  trainer_kwargs = trainer_kwargs or {}

  trainer = pl.Trainer(
    accelerator=accelerator,
    devices=devices,
    max_epochs=epochs,
    log_every_n_steps=10,
    logger=logger,
    callbacks=callbacks,
    # deterministic=True,
    **trainer_kwargs
  )
  trainer.fit(model, train_loader, val_loader)
  trainer.test(dataloaders=test_loader, ckpt_path='best')

  logger.experiment.finish()

  if not keep_local_ckpt_file_after_experiment_end:
    shutil.rmtree(CHECKPOINTS_DIR, ignore_errors=True)
    shutil.rmtree(LOGS_DIR, ignore_errors=True)
