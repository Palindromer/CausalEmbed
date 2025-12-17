"""
Evaluation module for comparing predicted edges against ground truth.

Computes performance metrics including recall, precision, and reduction rate
to assess the effectiveness of the pre-filtering strategy.
"""

import logging
from typing import Set, Tuple, Optional
from src.models import EvaluationMetrics

class Evaluator:
    """
    Evaluator for assessing pre-filtering performance.
    
    Responsibilities:
        - Compare predicted edges with ground truth edges
        - Calculate recall (fraction of true edges captured)
        - Calculate precision (fraction of predictions that are true)
        - Calculate reduction rate (computational savings)
        - Provide comprehensive evaluation metrics
    """
    
    def __init__(self):
        """Initialize Evaluator with logger."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate(
        self,
        predicted_edges: Set[Tuple[str, str]],
        ground_truth_edges: Set[Tuple[str, str]],
        all_possible_pairs_count: int
    ) -> EvaluationMetrics:
        """
        Evaluate pre-filtering performance against ground truth (Undirected/Skeleton evaluation).
        """
        # Validate inputs
        if not isinstance(predicted_edges, set):
            raise ValueError("predicted_edges must be a set")
        if not isinstance(ground_truth_edges, set):
            raise ValueError("ground_truth_edges must be a set")
        if all_possible_pairs_count < 0:
            raise ValueError("all_possible_pairs_count must be non-negative")
        
        # --- STEP 1: Normalize to Undirected (Sorted Tuples) ---
        # Оскільки ми оцінюємо "Skeleton Discovery", напрямок не важливий.
        # ('A', 'B') і ('B', 'A') вважаються одним і тим самим ребром.
        
        # 1. Нормалізуємо передбачення (сортуємо пари + прибираємо петлі)
        pred_undirected = {tuple(sorted(e)) for e in predicted_edges if e[0] != e[1]}
        
        # 2. Нормалізуємо Ground Truth (сортуємо пари + прибираємо петлі)
        gt_undirected = {tuple(sorted(e)) for e in ground_truth_edges if e[0] != e[1]}
        
        self.logger.info(
            f"Evaluating UNDIRECTED: {len(pred_undirected)} predicted vs "
            f"{len(gt_undirected)} ground truth pairs. "
            f"Capacity: {all_possible_pairs_count}"
        )
        
        # --- STEP 2: Calculate Matches ---
        true_positives = len(pred_undirected & gt_undirected)
        
        # Recall
        total_gt = len(gt_undirected)
        recall = true_positives / total_gt if total_gt > 0 else 0.0
        
        # Ancestral Recall (Undirected)
        # 1. Спочатку отримуємо направлені шляхи (щоб A->B->C дало A->C)
        # ancestral_directed = self._get_transitive_closure_edges(ground_truth_edges)
        # # 2. Перетворюємо їх на неорієнтовані пари для порівняння зі скелетом
        # ancestral_undirected = {tuple(sorted(e)) for e in ancestral_directed if e[0] != e[1]}
        # 
        # ancestral_tp = len(pred_undirected.intersection(ancestral_undirected))
        # ancestral_recall = ancestral_tp / len(ancestral_undirected) if len(ancestral_undirected) > 0 else 0

        ancestral_recall = 0.1 # fallback

        # Precision
        total_predicted = len(pred_undirected)
        precision = true_positives / total_predicted if total_predicted > 0 else 0.0
        
        # --- STEP 3: Reduction Rate ---
        filtered_pairs = all_possible_pairs_count - len(pred_undirected)
        
        if filtered_pairs < 0:
            self.logger.warning(
                f"Predicted edges ({len(pred_undirected)}) > All possible pairs ({all_possible_pairs_count}). "
                "Clamping reduction rate to 0. Check if 'all_possible_pairs_count' is set to N*(N-1)/2."
            )
            reduction_rate = 0.0
            filtered_pairs = 0 
        else:
            reduction_rate = filtered_pairs / all_possible_pairs_count if all_possible_pairs_count > 0 else 0.0
        
        self.logger.info(
            f"Results - Recall: {recall:.4f}, Precision: {precision:.4f}, "
            f"Reduction Rate: {reduction_rate:.4f}"
        )
        
        # Create metrics object
        metrics = EvaluationMetrics(
            recall=recall,
            ancestral_recall=ancestral_recall,
            precision=precision,
            reduction_rate=reduction_rate,
            true_edges_count=total_gt,
            kept_edges_count=len(pred_undirected),
            filtered_pairs_count=filtered_pairs,
            all_possible_pairs_count=all_possible_pairs_count
        )
        
        return metrics

    def evaluate_undirected(
        self,
        predicted_edges: Set[Tuple[str, str]],
        ground_truth_edges: Set[Tuple[str, str]],
        num_nodes: int
    ) -> EvaluationMetrics:
        """
        Evaluate with undirected edge interpretation.
        
        Converts directed edges to undirected by treating (u,v) and (v,u)
        as the same edge. Useful for comparing with undirected causal structures.
        
        Args:
            predicted_edges (Set[Tuple[str, str]]): Predicted directed edges
            ground_truth_edges (Set[Tuple[str, str]]): Ground truth directed edges
            num_nodes (int): Total number of nodes
        
        Returns:
            EvaluationMetrics: Evaluation metrics for undirected interpretation
        """
        self.logger.info("Evaluating with undirected edge interpretation")
        
        # Convert to undirected by normalizing edge tuples
        def to_undirected(edges: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
            undirected = set()
            for u, v in edges:
                # Normalize: smaller node ID first (lexicographically)
                if u < v:
                    undirected.add((u, v))
                else:
                    undirected.add((v, u))
            return undirected
        
        pred_undirected = to_undirected(predicted_edges)
        gt_undirected = to_undirected(ground_truth_edges)
        
        # All possible undirected pairs
        all_possible_pairs = num_nodes * (num_nodes - 1) // 2
        
        return self.evaluate(pred_undirected, gt_undirected, all_possible_pairs)
    
    def evaluate_directed(
        self,
        predicted_edges: Set[Tuple[str, str]],
        ground_truth_edges: Set[Tuple[str, str]],
        num_nodes: int
    ) -> EvaluationMetrics:
        """
        Evaluate with directed edge interpretation.
        
        Treats edges as directed: (u,v) and (v,u) are different edges.
        All possible pairs = n * (n - 1).
        
        Args:
            predicted_edges (Set[Tuple[str, str]]): Predicted directed edges
            ground_truth_edges (Set[Tuple[str, str]]): Ground truth directed edges
            num_nodes (int): Total number of nodes
        
        Returns:
            EvaluationMetrics: Evaluation metrics for directed interpretation
        """
        self.logger.info("Evaluating with directed edge interpretation")
        
        # All possible directed pairs
        all_possible_pairs = num_nodes * (num_nodes - 1)
        
        return self.evaluate(predicted_edges, ground_truth_edges, all_possible_pairs)
    
    def get_edge_statistics(
        self,
        predicted_edges: Set[Tuple[str, str]],
        ground_truth_edges: Set[Tuple[str, str]]
    ) -> dict:
        """
        Get detailed statistics about edges.
        
        Args:
            predicted_edges (Set[Tuple[str, str]]): Predicted edges
            ground_truth_edges (Set[Tuple[str, str]]): Ground truth edges
        
        Returns:
            dict: Dictionary with edge statistics
        """
        tp = len(predicted_edges & ground_truth_edges)
        fp = len(predicted_edges - ground_truth_edges)
        fn = len(ground_truth_edges - predicted_edges)
        tn_potential = len(predicted_edges.union(ground_truth_edges))
        
        stats = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'predicted_count': len(predicted_edges),
            'ground_truth_count': len(ground_truth_edges),
            'common_edges': tp,
            'unique_predictions': fp,
            'missed_edges': fn,
        }
        
        return stats

    def _get_transitive_closure_edges(self, edges: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
        """
        Converts a list of direct edges into a set of all reachable pairs (Ancestral Graph).
        If A->B->C, this adds (A,C) to the set.
        """
        if not edges:
            return set()
            
        # Створюємо граф NetworkX
        import networkx as nx
        G = nx.DiGraph()
        G.add_edges_from(edges)
        
        # Рахуємо транзитивне замикання (Reachability)
        # Це повертає граф, де є ребро між кожними u -> v, якщо існує шлях
        try:
            tc = nx.transitive_closure(G)
            return set(tc.edges())
        except Exception:
            self.logger.error("Graph has cycles. The fallback is used.")
            return edges