"""
LLM service module for semantic enrichment using Google Gemini API.

Handles batch enrichment of node names into semantic descriptions using
the Gemini API with robust caching to minimize API calls and costs.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import google.generativeai as genai

from src.config import Config


class LLMService:
    """
    Wrapper around Google Gemini API for text generation and node enrichment.
    
    Responsibilities:
        - Generate semantic descriptions for Bayesian Network nodes
        - Cache descriptions locally to minimize API calls
        - Handle batch processing for efficiency
        - Parse and validate JSON responses from Gemini
    """
    
    def __init__(self):
        """
        Initialize LLMService with API configuration.
        
        Args:
            api_key (str): Google Gemini API key.
        """
        self.config = Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        api_key = self.config.google_api_keys[0]        
        genai.configure(api_key=api_key)
        self.model = self.config.llm_model_name
        self.logger.info(f"LLMService initialized with model: {self.model}")
    
    def enrich_nodes(
        self,
        node_list: List[str],
        dataset_name: str,
        context_prompt: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Enrich node names with semantic descriptions using Gemini API.
        
        Implements robust caching: checks if descriptions already exist locally
        before making API calls. Uses batch processing to request all descriptions
        in a single API call for efficiency.
        
        Args:
            node_list (List[str]): List of node names/IDs to enrich
            dataset_name (str): Name of dataset (used for cache file naming)
            context_prompt (Optional[str]): Custom context for enrichment.
                If None, uses default causal discovery context.
        
        Returns:
            Dict[str, str]: Dictionary mapping node_id -> semantic description
        
        Raises:
            ValueError: If API response cannot be parsed as JSON
            Exception: If Gemini API call fails
        """
        self.logger.info(f"Enriching {len(node_list)} nodes for dataset: {dataset_name}")
        
        # Check cache first
        cache_file = self.config.cache_path / f"nodes_{dataset_name}.json"
        
        if cache_file.exists() and self.config.caching_enabled:
            self.logger.info(f"Loading cached node descriptions from {cache_file}")
            with open(cache_file, 'r') as f:
                descriptions = json.load(f)
            return descriptions

        # If not cached or incomplete, call API
        if context_prompt is None:
            context_prompt = self._get_domain_context(dataset_name)

        descriptions = self._call_gemini_for_descriptions(node_list, context_prompt)
        
        # Save to cache
        if self.config.caching_enabled:
            self.logger.info(f"Caching descriptions to {cache_file}")
            with open(cache_file, 'w') as f:
                json.dump(descriptions, f, indent=2)
        
        return descriptions
    
    def _call_gemini_for_descriptions(self, node_list: List[str], context_prompt: str) -> Dict[str, str]:

        nodes_str = ", ".join(node_list)
        prompt = self._build_prompt(context_prompt, nodes_str)
        
        response = genai.GenerativeModel(self.model).generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    def _build_prompt(self, context_prompt: str, nodes_str: str) -> str:
        return f"""
You are an expert in causal inference and Bayesian networks working in the domain: {context_prompt}.

TASK:
For each of the listed variable names, produce a concise, causally-informative, and *distinctive* description (1-4 sentences).
Do NOT use boilerplate sentence starters like "A variable indicating", "This node", or "The presence of" and other fillers. STRICTLY FORBIDDEN to start with: "A variable", "The ratio", "A metric", "The level", "Indicates", "Refers to", or "The presence of".
Do NOT repeat the same grammatical structure for each item.

Return a JSON object where keys are the variable names and values are the definitions.
Format: {{"variable_name": "scientific definition", ...}}
Return ONLY the JSON object.

Variables:
{nodes_str}
""".strip()

    def _get_domain_context(self, dataset_name: str) -> str:
        """
        Returns the specific scientific or technical domain for the dataset.
        Accurate context is crucial for LLM to generate correct semantic embeddings.
        """
        name = dataset_name.lower()

        # --- MEDICAL GROUP 1: General Diagnosis & Physiology ---
        if name in ['asia', 'cancer', 'alarm', 'child']:
            return (
                "Domain: Medical Diagnosis and Physiology. "
                "Context: Causal relationships between patient symptoms, diseases, risk factors, and clinical test results. "
            )
        elif name in ['hepar2']:
             return (
                "Domain: Hepatology and Liver Pathology. "
                "Context: Complex diagnosis of liver disorders (Steatosis, Cirrhosis, Hepatitis). "
                "Focus on the causal chain: Risk Factors (e.g., Alcohol) -> Pathology (e.g., Necrosis) -> "
                "Biomarkers (e.g., ALT, Bilirubin) -> Clinical Symptoms (e.g., Jaundice, Pain). "
                "Use standard medical abbreviations."
            )
        # --- MEDICAL GROUP 2: Specific Pathology ---
        elif 'pathfinder' in name:
            return (
                "Domain: Medical Histopathology (Lymph Nodes). "
                "Context: Diagnosis of lymph node diseases based on morphological features (F-nodes) observed in microscopy. "
                "Distinguishing benign changes from malignant lymphomas."
            )
        elif 'diabetes' in name:
            return (
                "Domain: Endocrinology and Metabolism. "
                "Context: Insulin dose adjustment, blood glucose regulation, and metabolic factors affecting diabetic patients."
            )
        elif 'munin' in name:
            # munin, munin1, munin2...
            return (
                "Domain: Neurology and Electromyography (EMG). "
                "Context: Diagnosis of neuromuscular diseases based on nerve conduction studies and muscle electrical activity measurements."
            )
            
        # --- BIOLOGY & GENETICS ---
        elif name in ['sachs']:
            return (
                "Domain: Molecular Biology (Cell Signaling). "
                "Context: Causal protein-signaling networks in human immune cells. Nodes represent phosphoproteins and enzymes."
            )
        elif name in ['link']:
            return (
                "Domain: Genetics (Linkage Analysis). "
                "Context: Genetic inheritance patterns, chromosomal linkage, and pedigree analysis for tracing traits."
            )
        elif name in ['pigs']:
            return (
                "Domain: Veterinary Science and Animal Breeding. "
                "Context: Pedigree analysis, genetic selection, and health traits in pig breeding programs."
            )

        # --- TECHNICAL & INDUSTRIAL ---
        elif name in ['win95pts']:
            return (
                "Domain: IT Technical Support and Troubleshooting. "
                "Context: Diagnosing printer and printing subsystem failures in the Windows 95 operating system. "
                "Nodes represent hardware states (Paper, Ink), driver settings, and OS errors."
            )
        elif name in ['water']:
            return (
                "Domain: Industrial Process Control (Water Treatment). "
                "Context: Physical and chemical processes in a wastewater treatment plant. "
                "Sensors monitoring flow, pressure, and chemical concentrations."
            )
        elif name in ['andes']:
            return (
                "Domain: Intelligent Tutoring Systems (Physics). "
                "Context: Assessing student knowledge states and strategy in solving physics problems (Newton's laws, vectors)."
            )

        # --- OTHER DOMAINS ---
        elif name in ['hailfinder']:
            return (
                "Domain: Meteorology and Severe Weather Forecasting. "
                "Context: Predicting severe weather (hail, tornadoes) in Colorado based on wind, moisture, and thermodynamic instability."
            )
        elif name in ['insurance']:
            return (
                "Domain: Actuarial Science and Risk Analysis. "
                "Context: Evaluating car insurance risk based on driver age, vehicle type, driving history, and accident costs."
            )
        elif name in ['earthquake']:
            return (
                "Domain: Seismology and Emergency Response. "
                "Context: Distinguishing between an earthquake and a burglary alarm based on radio reports and ground shaking."
            )
        elif name in ['barley']:
            return (
                "Domain: Agricultural Decision Support (Malting Barley). "
                "Context: Optimizing barley grain yield and protein quality for malt production. "
                "Managing nitrogen fertilization, fungicide application strategies, and controlling mildew or pest outbreaks."
            )

        # --- FALLBACK ---
        else:
            return "Domain: Causal Inference in Complex Systems. Context: Relationships between interacting variables."