"""
CausalEmbed: Embeddings-based Pre-filtering for Causal Discovery

This package provides a modular, production-ready solution for reducing
computational complexity of causal discovery algorithms through semantic
similarity-based pre-filtering of variable pairs.

Main components:
    - config: Configuration management (Singleton pattern)
    - models: Data classes and DTOs
    - data_loader: Bayesian Network dataset interaction
    - llm_service: Gemini API wrapper for text generation
    - embedding_service: Gemini API wrapper for embeddings
    - graph_processor: Similarity calculation and edge filtering
    - evaluator: Metrics computation against ground truth
"""

__version__ = "1.0.0"
__author__ = "Causal Research Team"

from src.config import Config
from src.models import CausalNode, EvaluationMetrics
from src.data_loader import DataLoader
from src.llm_service import LLMService
from src.embedding_service import EmbeddingService
from src.graph_processor import GraphProcessor
from src.evaluator import Evaluator
from src.results_manager import ResultsManager
from src.visualization import Visualizer
from src.pipeline import CausalPipeline

__all__ = [
    "Config",
    "CausalNode",
    "EvaluationMetrics",
    "DataLoader",
    "LLMService",
    "EmbeddingService",
    "GraphProcessor",
    "Evaluator",
    "ResultsManager",
    "Visualizer",
    "CausalPipeline",
]
