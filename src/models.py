"""
Data models and DTOs for the Causal Discovery pipeline.

This module contains dataclasses that represent the core data structures
used throughout the application, ensuring type safety and clarity.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CausalNode:
    """
    Represents a node in a Bayesian Network with semantic enrichment.
    
    Attributes:
        id (str): Unique identifier for the node (e.g., "tub").
        short_name (str): Short name or label (e.g., "Tub").
        description (str): Full semantic definition from LLM enrichment.
        embedding (Optional[np.ndarray]): Vector embedding of the description.
                                         Shape: (embedding_dim,).
    """
    id: str
    short_name: str
    description: str
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate that required fields are non-empty."""
        if not self.id or not self.short_name:
            raise ValueError("id and short_name cannot be empty")


@dataclass
class EvaluationMetrics:
    """
    Evaluation metrics for pre-filtering effectiveness.
    
    Attributes:
        recall (float): Fraction of ground truth edges captured by prediction.
                       recall = true_positives / total_ground_truth_edges
        precision (float): Fraction of predicted edges that are ground truth.
                          precision = true_positives / total_predicted_edges
        reduction_rate (float): Fraction of candidate pairs filtered out.
                               reduction_rate = 1 - (predicted_pairs / all_possible_pairs)
        true_edges_count (int): Total number of ground truth edges.
        kept_edges_count (int): Number of edges predicted by the filter.
        filtered_pairs_count (int): Number of pairs removed by filtering.
        all_possible_pairs_count (int): Total possible pairs (N choose 2).
    """
    recall: float
    precision: float
    reduction_rate: float
    true_edges_count: int
    kept_edges_count: int
    ancestral_recall: float = 0.0
    filtered_pairs_count: int = 0
    all_possible_pairs_count: int = 0

    def __post_init__(self):
        """Validate metric values are in reasonable ranges."""
        if not (0 <= self.recall <= 1):
            raise ValueError(f"Recall must be in [0, 1], got {self.recall}")
        if not (0 <= self.precision <= 1):
            raise ValueError(f"Precision must be in [0, 1], got {self.precision}")
        if not (0 <= self.reduction_rate <= 1):
            raise ValueError(f"Reduction rate must be in [0, 1], got {self.reduction_rate}")

    def __str__(self) -> str:
        """Pretty print metrics."""
        return (
            f"EvaluationMetrics(\n"
            f"  Recall: {self.recall:.4f}\n"
            f"  Ancestral Recall: {self.ancestral_recall:.4f}\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Reduction Rate: {self.reduction_rate:.4f}\n"
            f"  True Edges: {self.true_edges_count}\n"
            f"  Kept Edges: {self.kept_edges_count}\n"
            f"  Filtered Pairs: {self.filtered_pairs_count}\n"
            f"  All Possible Pairs: {self.all_possible_pairs_count}\n"
            f")"
        )
