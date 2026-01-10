"""Probe training and evaluation."""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.io import save_pickle, load_pickle, save_json, ensure_dir
from ..utils.stats import bootstrap_confidence_interval, compute_stratified_metrics


class ProbeExperiment:
    """
    Complete probe experiment: train, evaluate, visualize.
    """

    def __init__(
        self,
        activations_train: np.ndarray,
        labels_train: np.ndarray,
        activations_test: np.ndarray,
        labels_test: np.ndarray,
        metadata_train: Optional[Dict] = None,
        metadata_test: Optional[Dict] = None
    ):
        """
        Initialize probe experiment.

        Args:
            activations_train: Training activations (n_train, hidden_dim)
            labels_train: Training labels (n_train,)
            activations_test: Test activations (n_test, hidden_dim)
            labels_test: Test labels (n_test,)
            metadata_train: Optional training metadata (length_labels, groups, etc.)
            metadata_test: Optional test metadata
        """
        self.X_train = activations_train
        self.y_train = labels_train
        self.X_test = activations_test
        self.y_test = labels_test

        self.metadata_train = metadata_train or {}
        self.metadata_test = metadata_test or {}

        self.probe = None
        self.predictions = None
        self.metrics = None

        print(f"ProbeExperiment initialized:")
        print(f"  Train: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features")
        print(f"  Test: {self.X_test.shape[0]} samples")

    def train_probe(
        self,
        probe_type: str = 'logistic',
        regularization: str = 'cv',
        C: float = 1.0,
        cv_folds: int = 5,
        random_state: int = 42
    ) -> None:
        """
        Train linear probe.

        Args:
            probe_type: Type of probe ('logistic')
            regularization: 'fixed' (use C param) or 'cv' (cross-validation)
            C: Regularization parameter if regularization='fixed'
            cv_folds: Number of CV folds if regularization='cv'
            random_state: Random seed
        """
        print(f"\nTraining probe:")
        print(f"  Type: {probe_type}")
        print(f"  Regularization: {regularization}")

        if probe_type != 'logistic':
            raise ValueError(f"Unsupported probe type: {probe_type}")

        if regularization == 'cv':
            # Use cross-validation to select C
            self.probe = LogisticRegressionCV(
                Cs=[0.001, 0.01, 0.1, 1, 10, 100],
                cv=cv_folds,
                random_state=random_state,
                max_iter=1000,
                n_jobs=-1
            )
            self.probe.fit(self.X_train, self.y_train)
            print(f"  Best C (from CV): {self.probe.C_[0]:.4f}")

        else:  # fixed C
            self.probe = LogisticRegression(
                C=C,
                random_state=random_state,
                max_iter=1000
            )
            self.probe.fit(self.X_train, self.y_train)
            print(f"  C: {C}")

        # Compute train accuracy
        train_acc = self.probe.score(self.X_train, self.y_train)
        print(f"  Train accuracy: {train_acc:.4f}")

    def evaluate(
        self,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Evaluate probe on test set with bootstrap confidence intervals.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary of metrics
        """
        if self.probe is None:
            raise ValueError("No probe trained. Call train_probe() first.")

        print(f"\nEvaluating probe:")

        # Get predictions
        self.predictions = self.probe.predict(self.X_test)

        # Overall accuracy with confidence interval
        acc_mean, acc_lower, acc_upper = bootstrap_confidence_interval(
            self.y_test,
            self.predictions,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level
        )

        self.metrics = {
            'accuracy_overall': float(acc_mean),
            'accuracy_ci_lower': float(acc_lower),
            'accuracy_ci_upper': float(acc_upper)
        }

        print(f"  Overall accuracy: {acc_mean:.4f} [{acc_lower:.4f}, {acc_upper:.4f}]")

        # AUROC if probabilities available
        try:
            y_proba = self.probe.predict_proba(self.X_test)[:, 1]
            auroc = roc_auc_score(self.y_test, y_proba)
            self.metrics['auroc'] = float(auroc)
            print(f"  AUROC: {auroc:.4f}")
        except:
            pass

        # Stratified metrics if metadata available
        if 'length_labels' in self.metadata_test and 'sentiment_labels' in self.metadata_test:
            stratified = compute_stratified_metrics(
                self.y_test,
                self.predictions,
                np.array(self.metadata_test['length_labels']),
                np.array(self.metadata_test['sentiment_labels'])
            )
            self.metrics.update(stratified)

            print(f"\n  Stratified metrics:")
            for key, value in stratified.items():
                if key.startswith('accuracy_'):
                    print(f"    {key}: {value:.4f}")

            # Worst-group with bootstrap
            if 'accuracy_worst_group' in stratified:
                print(f"\n  Worst-group accuracy: {stratified['accuracy_worst_group']:.4f}")

        return self.metrics

    def visualize_results(
        self,
        save_path: Optional[Path] = None,
        show_plot: bool = False
    ) -> None:
        """
        Create visualizations of probe results.

        Args:
            save_path: Path to save figure
            show_plot: Whether to display plot
        """
        if self.metrics is None:
            raise ValueError("No metrics computed. Call evaluate() first.")

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Overall accuracy with CI
        ax1 = axes[0]
        self._plot_accuracy_bars(ax1)

        # Plot 2: Confusion matrix (if stratified metrics available)
        ax2 = axes[1]
        if 'length_labels' in self.metadata_test:
            self._plot_stratified_accuracy(ax2)
        else:
            self._plot_confusion_matrix(ax2)

        plt.tight_layout()

        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _plot_accuracy_bars(self, ax):
        """Plot accuracy with confidence intervals."""
        acc = self.metrics['accuracy_overall']
        ci_lower = self.metrics['accuracy_ci_lower']
        ci_upper = self.metrics['accuracy_ci_upper']
        error = [[acc - ci_lower], [ci_upper - acc]]

        ax.bar(['Overall'], [acc], yerr=error, capsize=10, color='steelblue', alpha=0.7)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1.0])
        ax.set_title('Overall Test Accuracy')
        ax.legend()

        # Add value labels
        ax.text(0, acc + 0.05, f'{acc:.3f}', ha='center', va='bottom')

    def _plot_stratified_accuracy(self, ax):
        """Plot per-group accuracies."""
        # Extract group accuracies
        group_metrics = {
            k: v for k, v in self.metrics.items()
            if k.startswith('accuracy_') and k not in ['accuracy_overall', 'accuracy_ci_lower', 'accuracy_ci_upper', 'accuracy_worst_group']
        }

        if not group_metrics:
            ax.text(0.5, 0.5, 'No stratified metrics', ha='center', va='center', transform=ax.transAxes)
            return

        # Sort by group name
        groups = sorted(group_metrics.keys())
        values = [group_metrics[g] for g in groups]

        # Clean up group names for display
        display_names = [g.replace('accuracy_', '').replace('_', '\n') for g in groups]

        # Plot bars
        bars = ax.bar(display_names, values, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:len(groups)], alpha=0.7)

        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.set_ylabel('Accuracy')
        ax.set_ylim([0, 1.0])
        ax.set_title('Per-Group Accuracy')
        ax.legend()

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    def _plot_confusion_matrix(self, ax):
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(self.y_test, self.predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

    def save(self, save_dir: Path, name: str = 'probe') -> None:
        """
        Save probe and results.

        Args:
            save_dir: Directory to save to
            name: Name prefix for saved files
        """
        save_dir = Path(save_dir)
        ensure_dir(save_dir)

        # Save probe model
        probe_path = save_dir / f"{name}_model.pkl"
        save_pickle(self.probe, probe_path)

        # Save predictions
        pred_path = save_dir / f"{name}_predictions.npy"
        np.save(pred_path, self.predictions)

        # Save metrics
        metrics_path = save_dir / f"{name}_metrics.json"
        save_json(self.metrics, metrics_path)

        print(f"✓ Saved probe experiment to {save_dir}")

    @classmethod
    def load(
        cls,
        save_dir: Path,
        name: str = 'probe',
        activations_train: np.ndarray = None,
        labels_train: np.ndarray = None,
        activations_test: np.ndarray = None,
        labels_test: np.ndarray = None
    ) -> 'ProbeExperiment':
        """
        Load saved probe experiment.

        Args:
            save_dir: Directory containing saved files
            name: Name prefix of saved files
            activations_train: Training activations (required for reinit)
            labels_train: Training labels
            activations_test: Test activations
            labels_test: Test labels

        Returns:
            Loaded ProbeExperiment instance
        """
        save_dir = Path(save_dir)

        # Load probe model
        probe_path = save_dir / f"{name}_model.pkl"
        probe = load_pickle(probe_path)

        # Create instance
        instance = cls(
            activations_train=activations_train,
            labels_train=labels_train,
            activations_test=activations_test,
            labels_test=labels_test
        )

        instance.probe = probe

        # Load predictions if available
        pred_path = save_dir / f"{name}_predictions.npy"
        if pred_path.exists():
            instance.predictions = np.load(pred_path)

        # Load metrics if available
        metrics_path = save_dir / f"{name}_metrics.json"
        if metrics_path.exists():
            from ..utils.io import load_json
            instance.metrics = load_json(metrics_path)

        return instance
