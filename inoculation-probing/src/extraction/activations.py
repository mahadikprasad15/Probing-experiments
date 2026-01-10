"""Activation extraction from language models."""

import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import hashlib
import json

from ..utils.io import save_numpy, load_numpy, save_json, load_json, ensure_dir


class ActivationExtractor:
    """
    Extract activations from specific layers of language models.

    Supports caching for efficient re-use of extracted activations.
    """

    def __init__(
        self,
        model_name: str = 'meta-llama/Llama-3.2-1B',
        layer: int = 12,
        device: str = 'auto',
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize activation extractor.

        Args:
            model_name: HuggingFace model name
            layer: Layer index to extract activations from
            device: Device to run on ('cuda', 'cpu', or 'auto')
            cache_dir: Directory to cache extracted activations
        """
        self.model_name = model_name
        self.layer = layer
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        print(f"Initializing ActivationExtractor:")
        print(f"  Model: {model_name}")
        print(f"  Layer: {layer}")
        print(f"  Device: {self.device}")

        # Load model and tokenizer
        print(f"Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine dtype
        if self.device == 'cuda':
            dtype = torch.float16
        elif self.device == 'mps':
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device if self.device == 'cuda' else None
        )

        if self.device != 'cuda':
            self.model = self.model.to(self.device)

        self.model.eval()

        # Get model config
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

        print(f"✓ Model loaded:")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Total layers: {self.num_layers}")

        if self.layer >= self.num_layers:
            raise ValueError(f"Layer {self.layer} exceeds model layers (0-{self.num_layers-1})")

    def _compute_cache_key(
        self,
        texts: List[str],
        prompt_name: str
    ) -> str:
        """Compute cache key for a set of texts and prompt."""
        # Create deterministic hash
        content = json.dumps({
            'model': self.model_name,
            'layer': self.layer,
            'prompt_name': prompt_name,
            'texts': texts  # Include texts for uniqueness
        }, sort_keys=True)

        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Optional[Path]:
        """Get cache file path."""
        if self.cache_dir is None:
            return None

        ensure_dir(self.cache_dir)
        return self.cache_dir / f"{cache_key}.npy"

    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Try to load activations from cache."""
        cache_path = self._get_cache_path(cache_key)

        if cache_path and cache_path.exists():
            print(f"✓ Loading from cache: {cache_path.name}")
            return load_numpy(cache_path)

        return None

    def _save_to_cache(self, cache_key: str, activations: np.ndarray) -> None:
        """Save activations to cache."""
        cache_path = self._get_cache_path(cache_key)

        if cache_path:
            save_numpy(activations, cache_path)

    def extract(
        self,
        texts: List[str],
        prompt_name: str = 'baseline',
        batch_size: int = 8,
        max_length: int = 512,
        use_cache: bool = True,
        position: str = 'last'
    ) -> np.ndarray:
        """
        Extract activations from texts.

        Args:
            texts: List of input texts
            prompt_name: Name of prompt (for caching)
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            use_cache: Whether to use caching
            position: Which token position to extract ('last', 'mean', 'max')

        Returns:
            Numpy array of shape (n_samples, hidden_size)
        """
        # Check cache
        cache_key = self._compute_cache_key(texts, prompt_name)

        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        # Extract activations
        print(f"Extracting activations from layer {self.layer}...")
        activations = []

        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract activations
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )

                # Get hidden states from specified layer
                # outputs.hidden_states is tuple of (num_layers + 1) tensors
                # Index 0 is embeddings, index 1 is layer 0, etc.
                layer_activations = outputs.hidden_states[self.layer + 1]  # +1 for embeddings

                # Extract based on position strategy
                if position == 'last':
                    # Get last non-padding token
                    batch_acts = self._extract_last_token(layer_activations, inputs['attention_mask'])
                elif position == 'mean':
                    batch_acts = self._extract_mean(layer_activations, inputs['attention_mask'])
                elif position == 'max':
                    batch_acts = self._extract_max(layer_activations, inputs['attention_mask'])
                else:
                    raise ValueError(f"Unknown position: {position}")

                activations.append(batch_acts.cpu().numpy())

        # Concatenate all batches
        activations = np.concatenate(activations, axis=0)

        print(f"✓ Extracted activations: shape={activations.shape}")

        # Save to cache
        if use_cache:
            self._save_to_cache(cache_key, activations)

        return activations

    def _extract_last_token(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract last non-padding token activations."""
        # Get sequence lengths (number of non-padding tokens)
        seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing

        # Extract last token for each sample
        batch_size = hidden_states.shape[0]
        last_token_acts = hidden_states[torch.arange(batch_size), seq_lengths]

        return last_token_acts

    def _extract_mean(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract mean-pooled activations."""
        # Mask out padding tokens
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        masked_hidden = hidden_states * mask

        # Mean over sequence length
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = mask.sum(dim=1)
        mean_hidden = sum_hidden / sum_mask

        return mean_hidden

    def _extract_max(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract max-pooled activations."""
        # Mask out padding tokens with large negative value
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        masked_hidden = hidden_states.clone()
        masked_hidden[mask == 0] = -1e9

        # Max over sequence length
        max_hidden, _ = masked_hidden.max(dim=1)

        return max_hidden

    def extract_multi_layer(
        self,
        texts: List[str],
        layers: List[int],
        prompt_name: str = 'baseline',
        batch_size: int = 8,
        max_length: int = 512
    ) -> Dict[int, np.ndarray]:
        """
        Extract activations from multiple layers.

        Args:
            texts: List of input texts
            layers: List of layer indices
            prompt_name: Name of prompt
            batch_size: Batch size for processing
            max_length: Maximum sequence length

        Returns:
            Dictionary mapping layer index to activations array
        """
        print(f"Extracting from {len(layers)} layers: {layers}")

        results = {}

        # Process all layers at once for efficiency
        activations_by_layer = {layer: [] for layer in layers}

        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )

                # Extract from each layer
                for layer in layers:
                    layer_acts = outputs.hidden_states[layer + 1]
                    batch_acts = self._extract_last_token(layer_acts, inputs['attention_mask'])
                    activations_by_layer[layer].append(batch_acts.cpu().numpy())

        # Concatenate each layer
        for layer in layers:
            results[layer] = np.concatenate(activations_by_layer[layer], axis=0)
            print(f"  Layer {layer}: shape={results[layer].shape}")

        return results

    def clear_cache(self) -> None:
        """Clear all cached activations."""
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.npy"):
                cache_file.unlink()
            print(f"✓ Cleared cache directory: {self.cache_dir}")
