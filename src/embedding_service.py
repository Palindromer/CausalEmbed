"""
Embedding service module for generating vector representations using Google Gemini API.

Handles generation and caching of semantic embeddings for node descriptions,
with support for batching, key rotation for rate limits, and multiple storage formats.
"""

import json
import logging
import time
import random
import itertools
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np

try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
except ImportError:
    raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")

from src.config import Config


class EmbeddingService:
    """
    Service for generating and managing text embeddings using Gemini API.
    Supports API Key Rotation to handle rate limits.
    """
    
    def __init__(self):
        """
        Initialize EmbeddingService with API configuration and Key Rotation.
        """
        self.config = Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.keys = self.config.google_api_keys
        if not self.keys:
            raise ValueError("No API keys found. Check your .env and Config.")

        self.logger.info(f"Loaded {len(self.keys)} API keys for rotation.")

        # 2. Setup Cycle Iterator
        self.key_cycle = itertools.cycle(self.keys)
        
        # 3. Initialize with the first key
        self._rotate_key(initial=True)
        
        self.embedding_model = self.config.embedding_model_name
        self.logger.info(f"EmbeddingService initialized with model: {self.embedding_model}")

    def _rotate_key(self, initial: bool = False):
        """Switches the active Gemini API key to the next one in the pool."""
        new_key = next(self.key_cycle)
        genai.configure(api_key=new_key)
        
        # Mask key for logging (show only last 4 chars)
        masked_key = f"...{new_key[-4:]}" if len(new_key) > 4 else "***"
        
        if initial:
            self.logger.info(f"Initialized active API key: {masked_key}")
        else:
            self.logger.warning(f"🔄 Switched API key to: {masked_key}")

    def generate_embeddings(
        self,
        text_dict: Dict[str, str],
        dataset_name: str,
        batch_size: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a dictionary of texts with caching.
        """
        self.logger.info(
            f"Generating embeddings for {len(text_dict)} texts from dataset: {dataset_name}"
        )
        
        # Determine cache file based on format
        cache_format = self.config.embeddings_cache_format
        if cache_format == 'npy':
            cache_file = self.config.cache_path / f"embeddings_{dataset_name}.npy"
            index_file = self.config.cache_path / f"embeddings_{dataset_name}_index.json"
        else:
            cache_file = self.config.cache_path / f"embeddings_{dataset_name}.pkl"
            index_file = None
        
        # Check cache first
        if cache_file.exists() and self.config.caching_enabled:
            self.logger.info(f"Loading cached embeddings from {cache_file}")
            try:
                if cache_format == 'npy':
                    embeddings_array = np.load(cache_file, allow_pickle=True).item()
                else:
                    import pickle
                    with open(cache_file, 'rb') as f:
                        embeddings_array = pickle.load(f)
                
                # Check if all keys are in cache
                if all(key in embeddings_array for key in text_dict.keys()):
                    self.logger.info("All embeddings found in cache")
                    return embeddings_array
                else:
                    self.logger.debug("Some embeddings missing from cache, will fetch from API")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}, will regenerate")
        
        # Generate embeddings if not cached or incomplete
        embeddings = self._generate_embeddings_batch(text_dict, batch_size)
        
        # Save to cache
        if self.config.caching_enabled:
            self.logger.info(f"Caching embeddings to {cache_file}")
            try:
                if cache_format == 'npy':
                    np.save(cache_file, embeddings, allow_pickle=True)
                    if index_file:
                        with open(index_file, 'w') as f:
                            json.dump(list(embeddings.keys()), f)
                else:
                    import pickle
                    with open(cache_file, 'wb') as f:
                        pickle.dump(embeddings, f)
            except Exception as e:
                self.logger.error(f"Failed to save cache: {e}")
        
        return embeddings
    
    def _generate_embeddings_batch(
        self,
        text_dict: Dict[str, str],
        batch_size: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings in batches with KEY ROTATION on Rate Limits (429).
        """
        embeddings = {}
        keys = list(text_dict.keys())
        texts = list(text_dict.values())
        
        num_batches = (len(texts) + batch_size - 1) // batch_size
        self.logger.info(f"Processing {len(texts)} texts in {num_batches} batches")
        
        # Increased retries because we have multiple keys to cycle through
        max_retries = 20
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(texts))
            
            batch_texts = texts[start_idx:end_idx]
            batch_keys = keys[start_idx:end_idx]
            
            self.logger.debug(f"Processing batch {batch_idx + 1}/{num_batches}")
            
            # --- RETRY & ROTATION LOGIC ---
            for attempt in range(max_retries):
                try:
                    # Call Gemini embedding API
                    result = genai.embed_content(
                        model=self.embedding_model,
                        content=batch_texts,
                        task_type="semantic_similarity"
                    )
                    
                    # Extract embeddings
                    batch_embeddings = result['embedding']
                    for key, embedding in zip(batch_keys, batch_embeddings):
                        embeddings[key] = np.array(embedding)
                    
                    # Success: exit retry loop
                    time.sleep(0.5) # Minimal pause to be nice to the API
                    break

                except Exception as e:
                    error_str = str(e).lower()
                    # Check for Rate Limit / Quota errors
                    if "429" in error_str or "quota" in error_str or "resource exhausted" in error_str:
                        self.logger.warning(
                            f"Quota hit (429) on batch {batch_idx+1}, attempt {attempt+1}. Rotating API Key..."
                        )
                        
                        # 1. Rotate to the next fresh key
                        self._rotate_key()
                        
                        # 2. Short wait to ensure switch propagates
                        time.sleep(2)
                        
                        # Loop continues -> retries immediately with new key
                    else:
                        # Non-recoverable error (e.g., Bad Request)
                        self.logger.error(f"Critical API Error: {e}")
                        raise e
            else:
                # Executes if loop finishes without 'break' (all retries failed)
                raise RuntimeError(
                    f"Failed to process batch {batch_idx} after {max_retries} attempts and key rotations."
                )
        
        self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this model.
        """
        self.logger.debug("Determining embedding dimension")
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content="test",
                task_type="semantic_similarity"
            )
            return len(result['embedding'])
        except Exception as e:
            self.logger.error(f"Failed to determine embedding dimension: {e}")
            raise