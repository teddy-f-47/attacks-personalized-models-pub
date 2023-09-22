from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()
STORAGE_DIR = PROJECT_DIR / 'storage'
CHECKPOINTS_DIR = STORAGE_DIR / 'checkpoints'
LOGS_DIR = STORAGE_DIR / 'logs'
PREDS_DUMP_DIR = STORAGE_DIR / 'preds'

SPLIT_NAMES = {
    'train': ['past', 'present'],
    'val': ['future1'],
    'test': ['future2']
}

EMBEDDINGS_SIZES = {
    'roberta': 768,
    'distilbert': 768
}

TRANSFORMER_MODEL_STRINGS = {
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-uncased'
}

GOEMOTIONS_LABELS = [
    'admiration',
    'amusement',
    'anger',
    'annoyance',
    'approval',
    'caring',
    'confusion',
    'curiosity',
    'desire',
    'disappointment',
    'disapproval',
    'disgust',
    'embarrassment',
    'excitement',
    'fear',
    'gratitude',
    'grief',
    'joy',
    'love',
    'nervousness',
    'optimism',
    'pride',
    'realization',
    'relief',
    'remorse',
    'sadness',
    'surprise',
    'neutral'
]

GOEMOTIONS_SENTIMENT_LABELS = ['neutral', 'positive', 'negative', 'ambiguous']

WIKI_DETOX_AGGRESSION_LABELS = ['aggression']
WIKI_DETOX_AGGRESSION_GENUINE_ANNOTATOR_IDS = PROJECT_DIR / 'storage' / 'wiki_detox_aggression_poisoned' / 'genuine_annotators.p'
WIKI_DETOX_AGGRESSION_SUBSET_GENUINE_ANNOTATOR_IDS = PROJECT_DIR / 'storage' / 'wiki_detox_aggression_subset_poisoned' / 'genuine_annotators.p'
