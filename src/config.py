"""
Configuration management module using Singleton pattern.

Loads configuration from both .env (environment variables) and config.yaml
(application settings). Ensures directories exist and provides centralized
access to all configuration parameters.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from dotenv import load_dotenv
from types import SimpleNamespace


class Config:
    """
    Singleton configuration manager for the Causal Discovery pipeline.
    
    Responsibilities:
        - Load environment variables from .env
        - Load application settings from config.yaml
        - Ensure all required directories exist
        - Provide centralized access to configuration parameters
    """
    
    _instance: Optional['Config'] = None
    
    def __new__(cls) -> 'Config':
        """Implement Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize configuration (only once due to Singleton pattern)."""
        if self._initialized:
            return
        
        self._initialized = True
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Get project root directory
        self.project_root = Path(__file__).parent.parent
        
        # Load environment variables from .env
        load_dotenv(self.project_root / '.env')
        
        keys_str = os.getenv("GOOGLE_API_KEYS")
        if not keys_str:
            raise ValueError("CRITICAL: Environment variable 'GOOGLE_API_KEYS' is missing. Please add comma-separated keys to .env")
            
        self.google_api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not self.google_api_keys:
            raise ValueError("CRITICAL: 'GOOGLE_API_KEYS' is present but empty.")
             
        # Set primary key (first one) for compatibility with simple calls
        self.logger.info(f"🔑 Loaded {len(self.google_api_keys)} Google API keys for rotation.")
        
        # Load YAML configuration
        config_path = self.project_root / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        paths_section = self._config.get('paths', {})
        self.paths = SimpleNamespace(
            raw=self.project_root / paths_section.get('raw', 'data/raw'),
            interim=self.project_root / paths_section.get('interim', 'data/interim'),
            processed=self.project_root / paths_section.get('processed', 'data/processed')
        )
        
        self.llm = SimpleNamespace(**self._config.get('llm', {}))
        self.filtering = SimpleNamespace(**self._config.get('filtering', {}))
        
        # Initialize directories
        self._initialize_directories()
        
        self.logger.info("Configuration initialized successfully")
    
    def _initialize_directories(self) -> None:
        """Create all required directories if they don't exist."""
        paths_config = self._config.get('paths', {})
        
        for path_key, path_value in paths_config.items():
            full_path = self.project_root / path_value
            full_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {full_path}")
    
    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self._config.get('llm', {})
    
    @property
    def llm_model_name(self) -> str:
        """Get LLM model name."""
        return self.llm_config.get('model_name')
    
    @property
    def embedding_model_name(self) -> str:
        """Get embedding model name."""
        return self.llm_config.get('embedding_model')
    
    @property
    def filtering_config(self) -> Dict[str, Any]:
        """Get filtering configuration."""
        return self._config.get('filtering', {})
    
    @property
    def default_threshold(self) -> float:
        """Get default similarity threshold for edge filtering."""
        return self.filtering_config.get('default_threshold')
    
    @property
    def top_k(self) -> int:
        """Get top-K parameter for rank-based filtering."""
        return self.filtering_config.get('top_k')
    
    @property
    def min_similarity(self) -> float:
        """Get minimum similarity threshold."""
        return self.filtering_config.get('min_similarity')
    
    @property
    def paths_config(self) -> Dict[str, str]:
        """Get all paths configuration."""
        return self._config.get('paths', {})
    
    def get_path(self, path_key: str) -> Path:
        """
        Get a configured path by key.
        
        Args:
            path_key (str): Key in paths configuration (e.g., 'raw', 'cache', 'results')
        
        Returns:
            Path: Full path object relative to project root
        """
        if path_key not in self.paths_config:
            raise KeyError(f"Path key '{path_key}' not found in configuration")
        
        return self.project_root / self.paths_config[path_key]
    
    @property
    def raw_data_path(self) -> Path:
        """Get path to raw data directory."""
        return self.get_path('raw')
    
    @property
    def cache_path(self) -> Path:
        """Get path to cache directory for intermediate data."""
        return self.get_path('cache')
    
    @property
    def results_path(self) -> Path:
        """Get path to results directory."""
        return self.get_path('results')
    
    @property
    def logs_path(self) -> Path:
        """Get path to logs directory."""
        return self.get_path('logs')
    
    @property
    def caching_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._config.get('caching', {}).get('enabled', True)
    
    @property
    def cache_format(self) -> str:
        """Get cache format (json or pickle)."""
        return self._config.get('caching', {}).get('format', 'json')
    
    @property
    def embeddings_cache_format(self) -> str:
        """Get embeddings cache format (npy or pickle)."""
        return self._config.get('caching', {}).get('embeddings_format', 'npy')
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        return self._config.copy()
