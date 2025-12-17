"""
Graph processing module for similarity calculation and edge filtering.

Handles computation of similarity matrices from embeddings and implements
both threshold-based and rank-based edge prediction strategies.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class GraphProcessor:
    """
    Processor for computing similarities and predicting causal edges.
    
    Responsibilities:
        - Calculate similarity matrices from embeddings
        - Implement threshold-based edge filtering
        - Implement rank-based edge filtering
        - Provide various similarity metrics (currently cosine)
        - Generate edge lists and statistics
    """
    
    def __init__(self):
        """Initialize GraphProcessor with logger."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_similarity_matrix(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        metric: str = 'cosine'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Calculate similarity matrix from embeddings.
        
        Computes pairwise similarity between all embeddings using the
        specified metric. Returns both as DataFrame and as array for efficiency.
        
        Args:
            embeddings_dict (Dict[str, np.ndarray]): Dictionary mapping node_id -> embedding
            metric (str): Similarity metric ('cosine' supported, default: 'cosine')
        
        Returns:
            Tuple[pd.DataFrame, List[str]]: Similarity matrix as DataFrame and node order
        
        Raises:
            ValueError: If metric not supported or embeddings invalid
        """
        if not embeddings_dict:
            raise ValueError("Embeddings dictionary is empty")
        
        self.logger.info(
            f"Calculating {metric} similarity matrix for {len(embeddings_dict)} nodes"
        )
        
        # Extract node IDs and embeddings in consistent order
        node_ids = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[node_id] for node_id in node_ids])
        
        # Validate embeddings
        if embeddings.shape[0] == 0:
            raise ValueError("No embeddings provided")
        
        if metric == 'cosine':
            # Compute cosine similarity
            similarity_matrix = cosine_similarity(embeddings)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Convert to DataFrame for easier indexing
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=node_ids,
            columns=node_ids
        )
        
        # Set diagonal to 0 (no self-edges)
        np.fill_diagonal(similarity_df.values, 0)
        
        self.logger.debug(
            f"Similarity matrix shape: {similarity_df.shape}, "
            f"mean similarity: {similarity_matrix[similarity_matrix != 0].mean():.4f}"
        )
        
        return similarity_df, node_ids
    
    def get_edges_by_threshold(
        self,
        similarity_matrix: pd.DataFrame,
        threshold: float,
        directed: bool = True
    ) -> Set[Tuple[str, str]]:
        """
        Predict edges based on similarity threshold.
        
        Threshold-based strategy: includes an edge (u, v) if similarity(u, v) > threshold.
        
        Args:
            similarity_matrix (pd.DataFrame): Similarity matrix from calculate_similarity_matrix
            threshold (float): Similarity threshold (must be in [0, 1])
            directed (bool): If True, considers edges as directed; if False, undirected
        
        Returns:
            Set[Tuple[str, str]]: Set of predicted edges as (source, target) tuples
        
        Raises:
            ValueError: If threshold not in valid range
        """
        if not (0 <= threshold <= 1):
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        
        self.logger.info(f"Extracting edges with threshold >= {threshold}")
        
        edges = set()
        
        # Iterate over upper triangle if undirected, all if directed
        if directed:
            # All pairs
            for i, source in enumerate(similarity_matrix.index):
                for j, target in enumerate(similarity_matrix.columns):
                    if i != j:  # No self-edges
                        similarity = similarity_matrix.iloc[i, j]
                        if similarity >= threshold:
                            edges.add((source, target))
        else:
            # Upper triangle (avoid duplicates)
            for i, source in enumerate(similarity_matrix.index):
                for j, target in enumerate(similarity_matrix.columns):
                    if i < j:  # Upper triangle
                        similarity = similarity_matrix.iloc[i, j]
                        if similarity >= threshold:
                            edges.add((source, target))
                            edges.add((target, source))
        
        self.logger.info(f"Found {len(edges)} edges with threshold {threshold}")
        return edges
    
    def get_edges_by_rank(self, similarity_matrix: pd.DataFrame, top_k: int) -> Set[Tuple[str, str]]:
        edges = set()
        for node in similarity_matrix.index:
            row = similarity_matrix.loc[node]
            
            # Remove self-loop if present.
            if node in row.index:
                row = row.drop(node)
                
            top_neighbors = row.nlargest(top_k).index.tolist()
            
            for neighbor in top_neighbors:
                # Sort the pair so that ('A', 'B') and ('B', 'A') become the same.
                # This ensures uniqueness of the pair regardless of direction.
                undirected_edge = tuple(sorted((node, neighbor)))
                edges.add(undirected_edge)
                
        return edges
    
    def get_edges_by_random(
        self,
        similarity_matrix: pd.DataFrame,
        count: int
    ) -> Set[Tuple[str, str]]:
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        
        node_ids = similarity_matrix.index.tolist()
        num_nodes = len(node_ids)
        
        max_possible_edges = num_nodes * (num_nodes - 1)
        effective_count = min(count, max_possible_edges)
        
        edges = set()
        while len(edges) < effective_count:
            source = np.random.choice(node_ids)
            target = np.random.choice(node_ids)
            if source != target:
                edges.add((source, target))

        return edges


    def get_edges_combined(
        self,
        similarity_matrix: pd.DataFrame,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> Set[Tuple[str, str]]:
        """
        Combine threshold and rank-based methods.
        
        Returns union of edges from both methods if both are specified,
        otherwise returns edges from the specified method.
        
        Args:
            similarity_matrix (pd.DataFrame): Similarity matrix
            threshold (Optional[float]): Similarity threshold
            top_k (Optional[int]): Top-K parameter
        
        Returns:
            Set[Tuple[str, str]]: Combined set of predicted edges
        """
        edges = set()
        
        if threshold is not None:
            edges.update(self.get_edges_by_threshold(similarity_matrix, threshold))
        
        if top_k is not None:
            edges.update(self.get_edges_by_rank(similarity_matrix, top_k))
        
        return edges
    
    def get_similarity_stats(self, similarity_matrix: pd.DataFrame) -> Dict:
        """
        Compute statistics about the similarity matrix.
        
        Args:
            similarity_matrix (pd.DataFrame): Similarity matrix
        
        Returns:
            Dict: Dictionary with statistical measures
        """
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        similarities = similarity_matrix.values[mask]
        
        stats = {
            'mean': float(np.mean(similarities)),
            'median': float(np.median(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'q25': float(np.percentile(similarities, 25)),
            'q75': float(np.percentile(similarities, 75)),
            'num_pairs': len(similarities),
        }
        
        return stats
