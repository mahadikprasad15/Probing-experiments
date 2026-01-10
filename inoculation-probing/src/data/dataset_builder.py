"""Dataset builder for creating spuriously correlated train/test splits."""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from ..utils.io import save_json, load_json


class SpuriousDatasetBuilder:
    """
    Creates train/test splits with controlled spurious correlations.

    Train set: Spurious correlation (e.g., short→positive, long→negative)
    Test set: Anti-correlated (e.g., short→negative, long→positive)
    """

    def __init__(
        self,
        source_dataset: str = 'imdb',
        train_correlation: float = 1.0,
        n_train_per_group: int = 100,
        n_test_per_group: int = 50,
        short_range: Tuple[int, int] = (10, 30),
        long_range: Tuple[int, int] = (100, 200),
        seed: int = 42
    ):
        """
        Initialize dataset builder.

        Args:
            source_dataset: HuggingFace dataset name ('imdb', 'sst2', 'yelp_polarity')
            train_correlation: Strength of spurious correlation (1.0 = perfect)
            n_train_per_group: Number of samples per group in train set
            n_test_per_group: Number of samples per group in test set
            short_range: Token count range for "short" reviews (min, max)
            long_range: Token count range for "long" reviews (min, max)
            seed: Random seed for reproducibility
        """
        self.source_dataset = source_dataset
        self.train_correlation = train_correlation
        self.n_train_per_group = n_train_per_group
        self.n_test_per_group = n_test_per_group
        self.short_range = short_range
        self.long_range = long_range
        self.seed = seed

        self.train_data = None
        self.test_data = None
        self.metadata = None

        np.random.seed(seed)

    def _load_source_data(self) -> Tuple[List[str], List[int]]:
        """Load source dataset from HuggingFace."""
        print(f"Loading {self.source_dataset} dataset...")

        if self.source_dataset == 'imdb':
            dataset = load_dataset('imdb', split='train')
            texts = dataset['text']
            labels = dataset['label']
        elif self.source_dataset == 'sst2':
            dataset = load_dataset('glue', 'sst2', split='train')
            texts = dataset['sentence']
            labels = dataset['label']
        elif self.source_dataset == 'yelp_polarity':
            dataset = load_dataset('yelp_polarity', split='train')
            texts = dataset['text']
            labels = dataset['label']
        else:
            raise ValueError(f"Unsupported dataset: {self.source_dataset}")

        print(f"✓ Loaded {len(texts)} samples")
        return texts, labels

    def _count_tokens(self, text: str) -> int:
        """Simple token count (split by whitespace)."""
        return len(text.split())

    def _filter_by_length(
        self,
        texts: List[str],
        labels: List[int],
        min_tokens: int,
        max_tokens: int
    ) -> Tuple[List[str], List[int]]:
        """Filter texts by token count range."""
        filtered_texts = []
        filtered_labels = []

        for text, label in zip(texts, labels):
            token_count = self._count_tokens(text)
            if min_tokens <= token_count <= max_tokens:
                filtered_texts.append(text)
                filtered_labels.append(label)

        return filtered_texts, filtered_labels

    def create_splits(self) -> Dict[str, Dict[str, List]]:
        """
        Create train/test splits with spurious correlation.

        Returns:
            Dictionary with 'train' and 'test' splits, each containing:
            - texts: List of text samples
            - sentiment_labels: List of sentiment labels (0=neg, 1=pos)
            - length_labels: List of length labels (0=short, 1=long)
            - groups: List of group names for stratification
        """
        texts, labels = self._load_source_data()

        # Separate by sentiment
        positive_texts = [t for t, l in zip(texts, labels) if l == 1]
        negative_texts = [t for t, l in zip(texts, labels) if l == 0]

        print(f"Positive samples: {len(positive_texts)}, Negative samples: {len(negative_texts)}")

        # Filter by length
        print(f"\nFiltering by length ranges:")
        print(f"  Short: {self.short_range[0]}-{self.short_range[1]} tokens")
        print(f"  Long: {self.long_range[0]}-{self.long_range[1]} tokens")

        short_pos, _ = self._filter_by_length(
            positive_texts, [1] * len(positive_texts),
            self.short_range[0], self.short_range[1]
        )
        long_pos, _ = self._filter_by_length(
            positive_texts, [1] * len(positive_texts),
            self.long_range[0], self.long_range[1]
        )
        short_neg, _ = self._filter_by_length(
            negative_texts, [0] * len(negative_texts),
            self.short_range[0], self.short_range[1]
        )
        long_neg, _ = self._filter_by_length(
            negative_texts, [0] * len(negative_texts),
            self.long_range[0], self.long_range[1]
        )

        print(f"\n✓ Filtered samples:")
        print(f"  Short+Positive: {len(short_pos)}")
        print(f"  Short+Negative: {len(short_neg)}")
        print(f"  Long+Positive: {len(long_pos)}")
        print(f"  Long+Negative: {len(long_neg)}")

        # Sample for train set (spurious correlation)
        # Train: short→positive, long→negative
        train_short_pos = self._sample(short_pos, self.n_train_per_group)
        train_long_neg = self._sample(long_neg, self.n_train_per_group)

        train_texts = train_short_pos + train_long_neg
        train_sentiment = [1] * len(train_short_pos) + [0] * len(train_long_neg)
        train_length = [0] * len(train_short_pos) + [1] * len(train_long_neg)
        train_groups = ['short_positive'] * len(train_short_pos) + ['long_negative'] * len(train_long_neg)

        # Sample for test set (anti-correlated)
        # Test: short→negative, long→positive
        test_short_neg = self._sample(short_neg, self.n_test_per_group)
        test_long_pos = self._sample(long_pos, self.n_test_per_group)

        test_texts = test_short_neg + test_long_pos
        test_sentiment = [0] * len(test_short_neg) + [1] * len(test_long_pos)
        test_length = [0] * len(test_short_neg) + [1] * len(test_long_pos)
        test_groups = ['short_negative'] * len(test_short_neg) + ['long_positive'] * len(test_long_pos)

        # Shuffle
        train_indices = np.random.permutation(len(train_texts))
        test_indices = np.random.permutation(len(test_texts))

        self.train_data = {
            'texts': [train_texts[i] for i in train_indices],
            'sentiment_labels': [train_sentiment[i] for i in train_indices],
            'length_labels': [train_length[i] for i in train_indices],
            'groups': [train_groups[i] for i in train_indices]
        }

        self.test_data = {
            'texts': [test_texts[i] for i in test_indices],
            'sentiment_labels': [test_sentiment[i] for i in test_indices],
            'length_labels': [test_length[i] for i in test_indices],
            'groups': [test_groups[i] for i in test_indices]
        }

        self.metadata = {
            'source_dataset': self.source_dataset,
            'train_correlation': self.train_correlation,
            'n_train_per_group': self.n_train_per_group,
            'n_test_per_group': self.n_test_per_group,
            'short_range': self.short_range,
            'long_range': self.long_range,
            'seed': self.seed,
            'train_size': len(self.train_data['texts']),
            'test_size': len(self.test_data['texts'])
        }

        print(f"\n✓ Created splits:")
        print(f"  Train size: {len(self.train_data['texts'])}")
        print(f"  Test size: {len(self.test_data['texts'])}")
        print(f"\nTrain distribution (spurious):")
        print(f"  Short+Positive: {sum(1 for g in self.train_data['groups'] if g == 'short_positive')}")
        print(f"  Long+Negative: {sum(1 for g in self.train_data['groups'] if g == 'long_negative')}")
        print(f"\nTest distribution (anti-correlated):")
        print(f"  Short+Negative: {sum(1 for g in self.test_data['groups'] if g == 'short_negative')}")
        print(f"  Long+Positive: {sum(1 for g in self.test_data['groups'] if g == 'long_positive')}")

        return {
            'train': self.train_data,
            'test': self.test_data,
            'metadata': self.metadata
        }

    def _sample(self, items: List, n: int) -> List:
        """Randomly sample n items without replacement."""
        if len(items) < n:
            raise ValueError(f"Not enough items to sample: requested {n}, available {len(items)}")
        indices = np.random.choice(len(items), size=n, replace=False)
        return [items[i] for i in indices]

    def save(self, save_dir: Path) -> None:
        """Save dataset splits and metadata."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.train_data is None or self.test_data is None:
            raise ValueError("No data to save. Call create_splits() first.")

        # Save splits
        save_json(self.train_data, save_dir / 'train.json')
        save_json(self.test_data, save_dir / 'test.json')
        save_json(self.metadata, save_dir / 'metadata.json')

        print(f"\n✓ Saved dataset to {save_dir}")

    @classmethod
    def load(cls, save_dir: Path) -> 'SpuriousDatasetBuilder':
        """Load dataset from saved files."""
        save_dir = Path(save_dir)

        metadata = load_json(save_dir / 'metadata.json')
        train_data = load_json(save_dir / 'train.json')
        test_data = load_json(save_dir / 'test.json')

        # Create instance with saved config
        instance = cls(
            source_dataset=metadata['source_dataset'],
            train_correlation=metadata['train_correlation'],
            n_train_per_group=metadata['n_train_per_group'],
            n_test_per_group=metadata['n_test_per_group'],
            short_range=tuple(metadata['short_range']),
            long_range=tuple(metadata['long_range']),
            seed=metadata['seed']
        )

        instance.train_data = train_data
        instance.test_data = test_data
        instance.metadata = metadata

        return instance
