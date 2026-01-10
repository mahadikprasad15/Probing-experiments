"""Logging utilities for experiment tracking."""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from .io import save_json, ensure_dir


def setup_logger(name: str, log_file: Optional[Path] = None, level=logging.INFO):
    """Setup logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        ensure_dir(log_file.parent)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ExperimentLogger:
    """Logger for tracking experiments with structured output."""

    def __init__(self, experiment_dir: Path):
        """
        Initialize experiment logger.

        Args:
            experiment_dir: Directory to save experiment logs
        """
        self.experiment_dir = Path(experiment_dir)
        ensure_dir(self.experiment_dir)

        self.runs = []
        self.current_run = None

        # Setup file logger
        self.logger = setup_logger(
            'ExperimentLogger',
            self.experiment_dir / 'experiment.log'
        )

    def start_run(self, run_name: str, config: Dict[str, Any]) -> str:
        """Start a new experimental run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{run_name}_{timestamp}"

        self.current_run = {
            'run_id': run_id,
            'run_name': run_name,
            'timestamp': timestamp,
            'config': config,
            'metrics': {},
            'artifacts': {}
        }

        self.logger.info(f"Started run: {run_id}")
        self.logger.info(f"Config: {json.dumps(config, indent=2)}")

        return run_id

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics for current run."""
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")

        self.current_run['metrics'].update(metrics)
        self.logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

    def log_artifact(self, artifact_name: str, artifact_path: Path) -> None:
        """Log artifact path for current run."""
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")

        self.current_run['artifacts'][artifact_name] = str(artifact_path)
        self.logger.info(f"Artifact '{artifact_name}' saved to: {artifact_path}")

    def end_run(self) -> None:
        """End current run and save results."""
        if self.current_run is None:
            raise ValueError("No active run to end.")

        self.runs.append(self.current_run)

        # Save individual run
        run_file = self.experiment_dir / f"{self.current_run['run_id']}.json"
        save_json(self.current_run, run_file)

        # Save all runs
        all_runs_file = self.experiment_dir / 'all_runs.json'
        save_json(self.runs, all_runs_file)

        self.logger.info(f"Ended run: {self.current_run['run_id']}")
        self.current_run = None

    def get_runs(self) -> list:
        """Get all logged runs."""
        return self.runs

    def load_runs(self) -> None:
        """Load previously logged runs."""
        all_runs_file = self.experiment_dir / 'all_runs.json'
        if all_runs_file.exists():
            with open(all_runs_file, 'r') as f:
                self.runs = json.load(f)
            self.logger.info(f"Loaded {len(self.runs)} previous runs")
