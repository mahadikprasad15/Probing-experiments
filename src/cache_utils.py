"""
Caching utilities for activations and directions.
Uses torch.save/load for efficient tensor storage.
"""
import os
import torch
import hashlib
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache"

def get_cache_path(name: str, suffix: str = ".pt") -> Path:
    """Returns the cache path for a given name."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{name}{suffix}"

def cache_exists(name: str) -> bool:
    """Check if cache exists."""
    return get_cache_path(name).exists()

def save_to_cache(data: dict, name: str) -> Path:
    """
    Save data dictionary to cache.
    Works with dicts containing torch tensors.
    """
    path = get_cache_path(name)
    torch.save(data, path)
    print(f"Cached: {path}")
    return path

def load_from_cache(name: str) -> dict:
    """Load data from cache."""
    path = get_cache_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    data = torch.load(path, weights_only=False)
    print(f"Loaded from cache: {path}")
    return data

def get_prompt_hash(prompts: list) -> str:
    """Generate a hash for a list of prompts for cache key."""
    content = "".join(prompts)
    return hashlib.md5(content.encode()).hexdigest()[:8]

def cache_activations(model, tokenizer, prompts, name_prefix: str, extractor_fn, force_recompute=False):
    """
    Wrapper that caches activation extraction.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        prompts: List of prompts
        name_prefix: Prefix for cache name (e.g., "harmful_acts")
        extractor_fn: Function(model, tokenizer, prompts) -> dict of activations
        force_recompute: If True, ignore cache and recompute
        
    Returns:
        dict: The activations (from cache or computed)
    """
    prompt_hash = get_prompt_hash(prompts)
    cache_name = f"{name_prefix}_{prompt_hash}"
    
    if not force_recompute and cache_exists(cache_name):
        return load_from_cache(cache_name)
    
    print(f"Computing {name_prefix}...")
    activations = extractor_fn(model, tokenizer, prompts)
    save_to_cache(activations, cache_name)
    return activations

def cache_directions(directions: dict, name: str, force_recompute=False):
    """Cache direction vectors."""
    if not force_recompute and cache_exists(name):
        return load_from_cache(name)
    save_to_cache(directions, name)
    return directions

def clear_cache():
    """Clear all cached files."""
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir()
        print("Cache cleared.")
