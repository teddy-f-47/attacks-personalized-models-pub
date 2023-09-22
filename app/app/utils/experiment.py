from sklearn.metrics import f1_score, accuracy_score
from torch import Tensor
import torchmetrics
import numpy as np


# METRICS
def evaluate_predictions(targets, predictions):
  targets_np = targets.cpu().detach().numpy()
  predictions_np = np.round(predictions.cpu().detach().numpy())

  acc = accuracy_score(y_true=targets_np, y_pred=predictions_np)
  f1 = f1_score(y_true=targets_np, y_pred=predictions_np, average='weighted')

  return acc, f1


class F1Class(torchmetrics.F1):
  def __init__(self, class_idx=-1, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.class_idx = class_idx

  def compute(self) -> Tensor:
    f1_for_classes = super().compute()
    return f1_for_classes[self.class_idx]


class PrecisionClass(torchmetrics.Precision):
  def __init__(self, class_idx=-1, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.class_idx = class_idx

  def compute(self) -> Tensor:
    precision_for_classes = super().compute()
    return precision_for_classes[self.class_idx]


class RecallClass(torchmetrics.Recall):
  def __init__(self, class_idx=-1, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.class_idx = class_idx

  def compute(self) -> Tensor:
    recall_for_classes = super().compute()
    return recall_for_classes[self.class_idx]
