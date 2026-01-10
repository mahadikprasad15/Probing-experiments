"""I/O utilities for saving and loading various file formats."""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Union[Dict, List], filepath: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    print(f"✓ Saved JSON to {filepath}")


def load_json(filepath: Union[str, Path]) -> Union[Dict, List]:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded JSON from {filepath}")
    return data


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """Save data to pickle file."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ Saved pickle to {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"✓ Loaded pickle from {filepath}")
    return data


def save_numpy(array: np.ndarray, filepath: Union[str, Path]) -> None:
    """Save numpy array to .npy file."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    np.save(filepath, array)
    print(f"✓ Saved numpy array to {filepath} (shape: {array.shape})")


def load_numpy(filepath: Union[str, Path]) -> np.ndarray:
    """Load numpy array from .npy file."""
    array = np.load(filepath)
    print(f"✓ Loaded numpy array from {filepath} (shape: {array.shape})")
    return array
