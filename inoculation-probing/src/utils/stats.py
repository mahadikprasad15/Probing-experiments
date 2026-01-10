"""Statistical utilities for analysis."""

import numpy as np
from typing import Dict, Tuple, Callable
from sklearn.utils import resample


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for a metric.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_fn: Metric function (default: accuracy)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if metric_fn is None:
        metric_fn = lambda y_t, y_p: np.mean(y_t == y_p)

    bootstrap_scores = []
    n_samples = len(y_true)

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = resample(np.arange(n_samples), n_samples=n_samples)
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)

    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    mean = np.mean(bootstrap_scores)
    lower = np.percentile(bootstrap_scores, lower_percentile)
    upper = np.percentile(bootstrap_scores, upper_percentile)

    return mean, lower, upper


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_labels: np.ndarray,
    group_names: Dict[int, str] = None
) -> Dict[str, float]:
    """
    Compute per-group accuracy and worst-group accuracy.

    Args:
        y_true: True labels (sentiment)
        y_pred: Predicted labels
        group_labels: Group identifiers (e.g., 0=short_pos, 1=short_neg, 2=long_pos, 3=long_neg)
        group_names: Optional mapping from group_labels to names

    Returns:
        Dictionary with per-group accuracies and worst-group accuracy
    """
    unique_groups = np.unique(group_labels)
    metrics = {}

    group_accuracies = []

    for group_id in unique_groups:
        group_mask = group_labels == group_id
        group_y_true = y_true[group_mask]
        group_y_pred = y_pred[group_mask]

        if len(group_y_true) == 0:
            continue

        accuracy = np.mean(group_y_true == group_y_pred)
        group_accuracies.append(accuracy)

        # Use group name if provided
        if group_names and group_id in group_names:
            group_key = group_names[group_id]
        else:
            group_key = f"group_{group_id}"

        metrics[f"accuracy_{group_key}"] = accuracy

    # Overall accuracy
    metrics['accuracy_overall'] = np.mean(y_true == y_pred)

    # Worst-group accuracy (minimum across all groups)
    if group_accuracies:
        metrics['accuracy_worst_group'] = min(group_accuracies)

    return metrics


def compute_stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    length_labels: np.ndarray,  # 0=short, 1=long
    sentiment_labels: np.ndarray  # 0=negative, 1=positive
) -> Dict[str, float]:
    """
    Compute metrics stratified by length and sentiment.

    Args:
        y_true: True sentiment labels
        y_pred: Predicted sentiment labels
        length_labels: Length category (0=short, 1=long)
        sentiment_labels: Sentiment category (0=negative, 1=positive)

    Returns:
        Dictionary with stratified accuracies
    """
    metrics = {}

    # Four groups: short_neg, short_pos, long_neg, long_pos
    groups = {
        'short_negative': (length_labels == 0) & (sentiment_labels == 0),
        'short_positive': (length_labels == 0) & (sentiment_labels == 1),
        'long_negative': (length_labels == 1) & (sentiment_labels == 0),
        'long_positive': (length_labels == 1) & (sentiment_labels == 1)
    }

    group_accuracies = []

    for group_name, mask in groups.items():
        if np.sum(mask) == 0:
            continue

        group_accuracy = np.mean(y_true[mask] == y_pred[mask])
        metrics[f"accuracy_{group_name}"] = group_accuracy
        group_accuracies.append(group_accuracy)

    # Overall and worst-group
    metrics['accuracy_overall'] = np.mean(y_true == y_pred)
    if group_accuracies:
        metrics['accuracy_worst_group'] = min(group_accuracies)

    return metrics
