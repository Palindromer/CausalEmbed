import pandas as pd
import logging
from datetime import datetime
from typing import Set, Tuple, Any, Dict, Optional
from pathlib import Path
import matplotlib.pyplot as plt

from .config import Config
from .models import EvaluationMetrics

class ResultsManager:
    """
    Handles persistence of experiment results, artifacts, and logs.
    """

    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.output_dir = self.config.paths.processed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a separate folder for plots to keep things organized
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def save_results(
        self,
        dataset_name: str,
        strategy_name: str,
        params: Dict[str, Any],
        metrics: EvaluationMetrics,
        similarity_matrix: pd.DataFrame,
        predicted_edges: Set[Tuple[str, str]],
        figure: Optional[plt.Figure] = None
    ) -> None:
        """
        Facade method to save all artifacts.
        Now accepts an optional Matplotlib Figure object.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self._save_experiment_log(dataset_name, strategy_name, params, metrics, timestamp)
        self._save_similarity_matrix(dataset_name, similarity_matrix)
        self._save_priors(dataset_name, predicted_edges)
        
        if figure:
            self._save_plot(dataset_name, strategy_name, figure)
        
        self.logger.info(f"✅ All results for '{dataset_name}' saved successfully.")

    def _save_plot(self, dataset_name: str, strategy_name: str, fig: plt.Figure) -> None:
        """Saves the matplotlib figure as a high-res PNG."""
        # Filename example: heatmap_asia_rank_top_k.png
        filename = self.plots_dir / f"plot_{dataset_name}_{strategy_name}.png"
        
        try:
            # bbox_inches='tight' removes extra whitespace around the plot
            # dpi=300 is standard for scientific publications
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.debug(f"Plot saved: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save plot: {e}")

    def _save_experiment_log(self, dataset_name: str, strategy_name: str, params: Dict[str, Any], metrics: EvaluationMetrics, timestamp: str) -> None:
        log_file = self.output_dir / "experiments_log.csv"
        param_str = "; ".join([f"{k}={v}" for k, v in params.items()])
        row_data = {
            "timestamp": timestamp,
            "dataset": dataset_name,
            "strategy": strategy_name,
            "parameters": param_str,
            "recall": round(metrics.recall, 4),
            "ancestral_recall": round(getattr(metrics, 'ancestral_recall', 0.0), 4),
            "precision": round(metrics.precision, 4),
            "reduction_rate": round(metrics.reduction_rate, 4),
            "true_edges": metrics.true_edges_count,
            "kept_edges": metrics.kept_edges_count
        }
        df = pd.DataFrame([row_data])
        if log_file.exists():
            df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            df.to_csv(log_file, mode='w', header=True, index=False)

    def _save_similarity_matrix(self, dataset_name: str, matrix: pd.DataFrame) -> None:
        filename = self.output_dir / f"similarity_{dataset_name}.csv"
        matrix.to_csv(filename)

    def _save_priors(self, dataset_name: str, edges: Set[Tuple[str, str]]) -> None:
        filename = self.output_dir / f"priors_{dataset_name}.csv"
        if not edges:
            pd.DataFrame(columns=['source', 'target']).to_csv(filename, index=False)
            return
        df = pd.DataFrame(list(edges), columns=['source', 'target'])
        df.to_csv(filename, index=False)