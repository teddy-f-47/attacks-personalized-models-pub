# code for wiki detox aggression subset (only texts with trigger), with distilBERT, fine-tuning

from itertools import product
import pickle
import torch
import os

from app.datasets.wiki_detox_aggression_subset_rev import WikiDetoxAggressionSubsetDataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from app.models import models as models_dict
from app.learning.train import train_test
from app.settings import LOGS_DIR, WIKI_DETOX_AGGRESSION_SUBSET_GENUINE_ANNOTATOR_IDS


torch.cuda.empty_cache()
seed_everything(47, workers=True)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WANDB_START_METHOD'] = 'thread'
os.makedirs(LOGS_DIR, exist_ok=True)

if __name__ == "__main__":
  wandb_project_name = 'WikipediaAggression_CP'
  keep_local_ckpt_file_after_experiment_end = False

  datamodule_cls = WikiDetoxAggressionSubsetDataModule
  regression = True
  poison_levels = [0.5]
  compromise_probs = [0, 0.25, 0.5, 0.75, 1.0]
  embeddings_types = ['distilbert']
  model_types = ['baseline_sgl', 'personalized_user_id', 'personalized_hubi_med']

  fold_nums = range(10)
  max_epochs = [3, 5]
  batch_size = 32
  lr_rates = [3e-5]

  user_folding = True
  use_cuda = True
  frozen = False

  test_only_clean_ann_group = True
  clean_ann_group = pickle.load(open(WIKI_DETOX_AGGRESSION_SUBSET_GENUINE_ANNOTATOR_IDS, "rb"))

  for (poison_level, compromise_prob, embeddings_type, lr_rate, model_type, mp) in product(poison_levels, compromise_probs, embeddings_types, lr_rates, model_types, max_epochs):
    major_voting = True if model_type=='baseline_avg' else False
    with_user_id_tokens = True if model_type=='personalized_user_id' else False
    epochs = mp

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
