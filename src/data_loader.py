import bnlearn
import numpy as np
import pandas as pd
import requests
import os
import gzip  # Added for decompressing .gz files
from typing import List, Tuple, Set, Dict, Any
import logging
from pathlib import Path

# Try importing pgmpy for direct reading as a fallback
try:
    from pgmpy.readwrite import BIFReader
except ImportError:
    BIFReader = None

from .config import Config

class DataLoader:
    """
    Handles interaction with the bnlearn library and file system.
    Responsible for downloading datasets and extracting ground truth structures.
    """
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.available_datasets = [
            'asia', 'cancer', 'earthquake', 'sachs', 'survey', 
            'alarm', 'barley', 'child', 'insurance', 'mildew', 
            'water', 'hailfinder', 'hepar2', 'win95pts'
        ]
        
        # Base URL for the official Bayesian Network Repository
        self.repo_url_base = "https://www.bnlearn.com/bnrepository"

    def _download_raw_file(self, dataset_name: str) -> Path:
        """
        Manually downloads and decompresses the .bif.gz file from bnlearn.com.
        """
        # Target local path: data/raw/hepar2.bif
        target_path = self.config.paths.raw / f"{dataset_name}.bif"
        
        if target_path.exists():
            return target_path

        # Construct URL: https://www.bnlearn.com/bnrepository/hepar2/hepar2.bif.gz
        url = f"{self.repo_url_base}/{dataset_name}/{dataset_name}.bif.gz"
        
        self.logger.info(f"Downloading and decompressing {dataset_name} from {url}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Decompress gzip content and write to file
            with gzip.GzipFile(fileobj=response.raw) as gz:
                content = gz.read()
                
            with open(target_path, 'wb') as f:
                f.write(content)
                
            self.logger.info(f"Successfully saved to {target_path}")
            return target_path
            
        except Exception as e:
            self.logger.error(f"Failed to download from bnlearn.com: {e}")
            # Fallback for small datasets (asia, cancer) which might not be in the big repo
            # or have different URL structure. But hepar2/alarm should work here.
            return None

    def download_and_load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Downloads (if necessary) and loads a dataset.
        Prioritizes manual download from bnlearn.com for stability, 
        then attempts library load.
        """
        model = None
        
        # Strategy 1: Check/Download local file first (Most robust for hepar2/alarm)
        local_file = self._download_raw_file(dataset_name)
        
        # Strategy 2: If we have a local file, try to load it
        if local_file and local_file.exists():
            try:
                # Try loading local file via bnlearn
                model = bnlearn.import_DAG(str(local_file), verbose=0)
            except Exception as e:
                self.logger.warning(f"bnlearn failed to load local file: {e}")
                # Try pgmpy direct reader
                if BIFReader:
                    self.logger.info("Falling back to pgmpy BIFReader...")
                    try:
                        reader = BIFReader(str(local_file))
                        model = reader.get_model()
                    except Exception as pgmpy_e:
                        self.logger.error(f"pgmpy failed too: {pgmpy_e}")

        # Strategy 3: If manual download failed, try bnlearn built-in downloader
        if model is None:
            self.logger.info(f"Trying bnlearn built-in download for '{dataset_name}'...")
            try:
                model = bnlearn.import_DAG(dataset_name, verbose=0)
            except Exception:
                pass

        # Check if we got anything valid
        is_empty = False
        if isinstance(model, dict) and not model:
            is_empty = True
        elif model is None:
            is_empty = True

        if is_empty:
             raise ValueError(f"Could not load dataset '{dataset_name}'. "
                              f"Manual download failed and built-in loader returned empty.")

        # --- Process the loaded model object into an Adjacency Matrix ---
        
        adj_matrix = None

        # Case A: Dictionary with 'adjmat' (Standard bnlearn)
        if isinstance(model, dict) and 'adjmat' in model:
            adj_matrix = model['adjmat']
            
        # Case B: Dictionary with 'model' (bnlearn large graphs)
        elif isinstance(model, dict) and 'model' in model:
            import networkx as nx
            nx_graph = model['model']
            adj_matrix = pd.DataFrame(
                nx.to_numpy_array(nx_graph), 
                columns=list(nx_graph.nodes()), 
                index=list(nx_graph.nodes())
            )

        # Case C: Pgmpy BayesianNetwork object (from BIFReader)
        elif hasattr(model, 'nodes') and hasattr(model, 'edges'):
             import networkx as nx
             # Convert pgmpy model to NetworkX DiGraph
             if hasattr(model, 'to_directed'): 
                 nx_graph = model
             else:
                 nx_graph = nx.DiGraph(model.edges())
                 nx_graph.add_nodes_from(model.nodes())
                 
             adj_matrix = pd.DataFrame(
                nx.to_numpy_array(nx_graph), 
                columns=list(nx_graph.nodes()), 
                index=list(nx_graph.nodes())
            )

        # Final Validation and Type Casting
        if adj_matrix is not None:
            try:
                adj_matrix = adj_matrix.astype(float).astype(int)
            except ValueError:
                adj_matrix = adj_matrix.astype(bool).astype(int)
                
            nodes = list(adj_matrix.columns)
            return adj_matrix, nodes
        
        # Debug info if extraction failed
        debug_info = type(model)
        if isinstance(model, dict):
            debug_info = f"Dict keys: {model.keys()}"
            
        raise ValueError(f"Could not extract adjacency matrix for '{dataset_name}'. Got: {debug_info}")

    def get_ground_truth_edges(self, dataset_name: str) -> Set[Tuple[str, str]]:
        adj_matrix, nodes = self.download_and_load_dataset(dataset_name)
        edges = set()
        for source in nodes:
            for target in nodes:
                if adj_matrix.loc[source, target] == 1: 
                    edges.add((source, target))
                    
        return edges

    def get_node_list(self, dataset_name: str) -> List[str]:
        _, nodes = self.download_and_load_dataset(dataset_name)
        return nodes

    def get_adjacency_matrix(self, dataset_name: str) -> pd.DataFrame:
        adj, _ = self.download_and_load_dataset(dataset_name)
        return adj
    
    def list_available_datasets(self) -> List[str]:
        return self.available_datasets
    
    def get_dataset_stats(self, dataset_name: str) -> Dict[str, int]:
        adj, nodes = self.download_and_load_dataset(dataset_name)
        edges = self.get_ground_truth_edges(dataset_name)
        possible = len(nodes) * (len(nodes)-1)
        return {
            "nodes": len(nodes),
            "edges": len(edges),
            "density": round(len(edges)/possible, 4) if possible > 0 else 0
        }