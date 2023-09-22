# code for goemo sentiment subset exps with distilBERT, fine-tuning

from itertools import product
import torch
import os

from app.datasets.goemotions_sentiment_subsetD_rev import GoEmotionsSentimentSubsetDataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from app.models import models as models_dict
from app.learning.train import train_test
from app.settings import LOGS_DIR


torch.cuda.empty_cache()
seed_everything(47, workers=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.makedirs(LOGS_DIR, exist_ok=True)

if __name__ == "__main__":
  wandb_project_name = 'GoEmotions_Sentiment_CP'
  keep_local_ckpt_file_after_experiment_end = False

  datamodule_cls = GoEmotionsSentimentSubsetDataModule
  regression = True
  poison_levels = [0.5]
  compromise_probs = [0, 0.25, 0.5, 0.75, 1.0]
  embeddings_types = ['distilbert']
  model_types = ['baseline_sgl', 'personalized_user_id', 'personalized_hubi_med']

  fold_nums = range(10)
  epochs = 5
  batch_size = 16
  lr_rates = [5e-5]

  user_folding = True
  use_cuda = True
  frozen = False

  test_only_clean_ann_group = True
  clean_ann_group = [0, 1, 2, 4, 7, 8, 10, 12, 15, 16, 18, 19, 20, 21, 25, 26, 29, 31, 33, 34, 35, 36, 38, 42, 43, 47, 49, 50, 51, 53, 55, 57, 58, 65, 68, 69, 74, 75, 77, 78, 79]

  for (poison_level, compromise_prob, embeddings_type, lr_rate, model_type) in product(poison_levels, compromise_probs, embeddings_types, lr_rates, model_types):
    major_voting = True if model_type=='baseline_avg' else False
    with_user_id_tokens = True if model_type=='personalized_user_id' else False

    data_module = datamodule_cls(
      poison_level=poison_level,
      compromise_probability=compromise_prob,
      folds=len(fold_nums),
      past_annotations_limit=None,
      major_voting=major_voting,
      normalize=regression,
      classification=(not regression),
      embeddings_type=embeddings_type,
      with_user_id_tokens=with_user_id_tokens,
      batch_size=batch_size,
      test_only_clean_ann_group=test_only_clean_ann_group,
      clean_ann_group=clean_ann_group
    )
    data_module.prepare_data()
    data_module.setup()
    vocab_size = data_module.current_vocab_size

    for fold_num in fold_nums:
      hparams = {
        "classification": (not regression),
        "learning_rate": lr_rate,
        "major_voting": major_voting,
        "with_user_id_tokens": with_user_id_tokens,
        "dataset": type(data_module).__name__,
        "poison_level": poison_level,
        "compromise_prob": compromise_prob,
        "embeddings_type": embeddings_type,
        "model_type": model_type,
        "frozen": frozen,
        "fold_num": fold_num,
        "max_epochs": epochs,
        "test_only_clean_ann_group": test_only_clean_ann_group
      }

      logger = pl_loggers.WandbLogger(
        save_dir=LOGS_DIR,
        config=hparams,
        project=wandb_project_name,
        log_model=False
      )

      output_dim = len(data_module.class_dims) if regression else sum(data_module.class_dims)
      model_cls = models_dict[model_type]

      model = model_cls(
        output_dim=output_dim,
        bert_layer_model_name=embeddings_type,
        vocab_size=vocab_size,
        word_num=data_module.words_number,
        annotator_num=data_module.annotators_number,
        bias_vector_length=len(data_module.class_dims)
      )

      if frozen:
        model.freeze_bert()
      else:
        model.unfreeze_bert()

      train_test(
        data_module,
        test_fold=fold_num,
        model=model,
        epochs=epochs,
        lr=lr_rate,
        regression=regression,
        use_cuda=use_cuda,
        logger=logger,
        keep_local_ckpt_file_after_experiment_end=keep_local_ckpt_file_after_experiment_end
      )
