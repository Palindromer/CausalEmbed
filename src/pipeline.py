import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import Config
from .data_loader import DataLoader
from .llm_service import LLMService
from .embedding_service import EmbeddingService
from .graph_processor import GraphProcessor
from .evaluator import Evaluator
from .results_manager import ResultsManager
from .visualization import Visualizer

class CausalPipeline:
    """
    Orchestrates the entire Causal Discovery via Embeddings pipeline.
    Supports running single datasets or batch processing multiple datasets.
    """

    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize services once
        self.loader = DataLoader()
        self.llm_service = LLMService()
        self.embed_service = EmbeddingService()
        self.processor = GraphProcessor()
        self.evaluator = Evaluator()
        self.saver = ResultsManager()

    def run_batch(self, dataset_names: List[str]) -> pd.DataFrame:
        """
        Runs the pipeline for multiple datasets and aggregates results.
        """
        all_results = []
        
        print(f"🚀 Starting Batch Processing for {len(dataset_names)} datasets...")
        
        for name in tqdm(dataset_names, desc="Datasets"):
            try:
                # Run pipeline for one dataset
                df_dataset = self.run_single_dataset(name, save_plots=True)
                
                # Add dataset-level metadata for global analytics
                stats = self.loader.get_dataset_stats(name)
                df_dataset['n_nodes'] = stats['nodes']
                df_dataset['density'] = stats['density']
                df_dataset['dataset'] = name
                
                all_results.append(df_dataset)
                
            except Exception as e:
                self.logger.error(f"❌ Failed processing dataset '{name}': {e}", exc_info=True)
                print(f"Skipping {name} due to error.")

        # Combine all results into one master DataFrame
        if all_results:
            master_df = pd.concat(all_results, ignore_index=True)
            
            # Save the master log
            master_file = self.config.paths.processed / "batch_results_summary.csv"
            master_df.to_csv(master_file, index=False)
            print(f"✅ Batch processing complete. Master log saved to {master_file}")
            return master_df
        else:
            return pd.DataFrame()

    def run_single_dataset(self, dataset_name: str, save_plots: bool = True) -> pd.DataFrame:
        """
        Runs the full pipeline for a single dataset:
        Load -> Embed -> Similarity -> Rank (Loop k) -> Evaluate -> Save.
        """
        self.logger.info(f"--- Processing {dataset_name} ---")

        # 1. Load Data
        adj_matrix, nodes = self.loader.download_and_load_dataset(dataset_name)
        ground_truth_edges = self.loader.get_ground_truth_edges(dataset_name)
        
        # Calculate Undirected Capacity for Evaluation
        n_nodes = len(nodes)
        all_possible_pairs = n_nodes * (n_nodes - 1) // 2
        
        # 2. Process Semantics (Descriptions + Embeddings)
        # Context is auto-detected inside llm_service now
        descriptions = self.llm_service.enrich_nodes(nodes, dataset_name)
        embeddings = self.embed_service.generate_embeddings(descriptions, dataset_name)
        
        # 3. Calculate Similarity
        sim_matrix, _ = self.processor.calculate_similarity_matrix(embeddings)
        
        # 4. Sensitivity Analysis (Loop over k)
        results = []
        
        k_range = range(1, n_nodes, 6)
        # Using tqdm for k-loop if it's large, otherwise silent
        iterator = tqdm(k_range, desc=f"Analyzing {dataset_name}", leave=False) if len(k_range) > 5 else k_range
        
        for k in iterator:
            # A. Filter
            predicted_edges = self.processor.get_edges_by_rank(sim_matrix, top_k=k)
            
            # B. Evaluate
            metrics = self.evaluator.evaluate(
                predicted_edges=predicted_edges,
                ground_truth_edges=ground_truth_edges,
                all_possible_pairs_count=all_possible_pairs
            )
            
            results.append({
                "k": k,
                "recall": metrics.recall,
                "ancestral_recall": getattr(metrics, 'ancestral_recall', 0),
                "precision": metrics.precision,
                "reduction_rate": metrics.reduction_rate
            })
            
        df_results = pd.DataFrame(results)
        
        # 5. Save Artifacts
        if save_plots:
            # Plot Sensitivity Graph
            fig = Visualizer.plot_k_sensitivity(df_results, dataset_name)
            
            # Save everything
            self.saver.save_results(
                dataset_name=dataset_name,
                strategy_name="batch_rank_k",
                params={"k_min": min(k_range), "k_max": max(k_range)},
                metrics=metrics, # Metrics of the last k (just for log structure)
                similarity_matrix=sim_matrix,
                predicted_edges=set(), # Not saving edges for every k to save space
                figure=fig
            )
            plt.close(fig) # Cleanup memory

        return df_results