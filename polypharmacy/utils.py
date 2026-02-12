import json
import os
import random
from typing import Any, Dict

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - fallback for environments without torch
    torch = None


def set_seeds(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, object]:
    """Compute ROC-AUC, accuracy, confusion matrix, F1, sens/spec from sigmoid outputs."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        recall_score,
        roc_auc_score,
    )

    preds = (probs >= 0.5).astype(np.int64)
    metrics: Dict[str, object] = {"accuracy": float(accuracy_score(labels, preds))}
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    metrics["sensitivity"] = float(recall_score(labels, preds, pos_label=1))
    metrics["specificity"] = float(recall_score(labels, preds, pos_label=0))
    metrics["f1"] = float(f1_score(labels, preds))
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    metrics["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics
