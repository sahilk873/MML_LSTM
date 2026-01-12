import json
from typing import Any, Dict, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 13,
    "train_frac": 0.8,
    "val_frac": 0.1,
    "test_frac": 0.1,
    "embedding_dim": 128,
    "lstm_hidden_dim": 128,
    "mlp_hidden_dim": 128,
    "dropout": 0.2,
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "freeze_kg": True,
    "kg_walk_length": 20,
    "kg_num_walks": 10,
    "kg_p": 1.0,
    "kg_q": 1.0,
    "kg_context_window": 10,
    "kg_min_count": 1,
    "kg_workers": 4,
}


def load_config(path: Optional[str]) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            overrides = json.load(handle)
        config.update(overrides)
    return config
