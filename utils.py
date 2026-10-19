"""utils.py – JSON helpers, logging setup, seed setting, and misc profiling utilities."""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from config import LOG_DIR


# ──────────────────────────────────────────────────
# JSON helpers
# ──────────────────────────────────────────────────

def save_json(data: Any, path: Union[str, Path]) -> None:
    """Serialize *data* to a JSON file at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    logging.info("Saved JSON → %s", path)


def load_json(path: Union[str, Path]) -> Any:
    """Load and return parsed JSON from *path*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    logging.info("Loaded JSON ← %s", path)
    return data


def _json_default(obj):
    """Fallback serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ──────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────

def setup_logging(
    name: str = "grems",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a named logger; optionally mirrors output to a file in LOG_DIR."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False

    # Optional file handler
    if log_file:
        fh = logging.FileHandler(LOG_DIR / log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info("Random seed set to %d", seed)


# ──────────────────────────────────────────────────
# Profiling helpers
# ──────────────────────────────────────────────────

def compute_flops_from_cfg(layer_cfg: tuple) -> int:
    """Return the estimated FLOPs for a layer config tuple (last element)."""
    return int(layer_cfg[-1])


def compute_activation_size(layer_cfg: tuple) -> int:
    """Return the activation tensor size (number of elements) for a layer config tuple (index 4)."""
    return int(layer_cfg[4])


def layer_type(layer_cfg: tuple) -> str:
    """Return the human-readable layer type string."""
    type_map = {"C": "Conv2d", "M": "MaxPool2d", "D": "Linear"}
    return type_map.get(layer_cfg[0], "Unknown")


# ──────────────────────────────────────────────────
# Misc
# ──────────────────────────────────────────────────

def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """Flatten a nested dictionary using *sep* as key separator."""
    items: List = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def moving_average(values: List[float], window: int = 10) -> List[float]:
    """Compute a simple moving average over *values*."""
    if len(values) < window:
        return values
    cumsum = np.cumsum(values, dtype=float)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    return (cumsum[window - 1 :] / window).tolist()
