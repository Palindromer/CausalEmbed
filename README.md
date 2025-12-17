# Hybrid Causal Discovery via LLM-based Semantic Filtering

**Author:** Oleksii Kovenko  
**Affiliation:** Kyiv Aviation Institute  
**Research Topic:** Information technology of hybrid causal risk analysis with LLM-priors.

---

## 🔬 Overview

This research addresses the **"Curse of Dimensionality"** in Causal Discovery. Classic algorithms (like PC, GES) struggle with exponential complexity when processing large-scale networks ($N > 100$ nodes), often requiring supercomputers or failing to converge due to the vast search space ($O(2^N)$).

We propose a **Semantic Pre-filtering Method** that utilizes Large Language Models (LLMs) to construct a "Sparsity Prior" (Skeleton). By analyzing the semantic meaning of variables, we filter out statistically unlikely connections before the causal search begins.

### 🧠 Core Hypothesis

Causal relationships are highly correlated with semantic similarity. By filtering out semantically distant pairs, we can reduce the search space for causal algorithms while preserving the True Positive causal edges (Recall).

#### Scalability via Semantic Sparsity

Unlike traditional statistical methods that degrade with size, we hypothesize that **our method's efficiency increases with the graph's complexity**.

1. **Small Graphs (Dense Semantics):** In small networks (e.g., 20 nodes), variables are often tightly related (e.g., all are symptoms of one disease). Semantic embeddings are clustered closely, making filtering harder.
2. **Large Graphs (Sparse Semantics):** In massive networks (e.g., 1000+ nodes), the system consists of distinct semantic modules (e.g., *Nerves* vs. *Muscles* vs. *Bones*). Even with a larger $k$ (neighbors), the ratio of filtered pairs increases significantly.

---

## ⚙️ Methodology

The fundamental mechanism of this study relies on the **Top-K Similarity Strategy**.

To ensure standardized comparison across datasets of vastly different sizes (from 30 to 1000+ nodes), we analyze the hyperparameter **$k$** (neighbors retained) relative to the total number of nodes ($N$). 

* **Relative $k$ & Reduction Rate:** Instead of a fixed integer, $k$ serves as a dynamic threshold (e.g., "Keep Top-5%"). This allows us to fix the **Reduction Rate** (e.g., filtering out 90% of pairs) and observe how Recall changes as the graph scales.

### The Research Pipeline

1.  **Data Ingestion:** * Raw Bayesian Networks are parsed from `.bif` format to extract nodes and ground truth structures.

2.  **Semantic Enrichment:**
    * **Model:** `gemini-3-flash-preview`.
    * **Process:** We inject a **Domain Context Prompt** to prevent hallucinations (e.g., distinguishing medical "flow" from engineering "flow"). The LLM generates a precise natural language definition for each node.

3.  **Vectorization:**
    * **Model:** `gemini-embedding-001`.
    * **Process:** Definitions are converted into high-dimensional semantic vectors.

4.  **Ranking & Filtering (Adaptive Top-K):**
    * We compute the **Cosine Similarity** matrix for all node pairs.
    * For each node, we retain the top $k$ neighbors, where $k$ is determined by the target Reduction Rate for the specific graph size.
    * This forms the **Predicted Skeleton**.

5.  **Evaluation:**
    * The Predicted Skeleton is compared against the **Ground Truth** (undirected), calculating **Recall** and **Reduction Rate** to assess the trade-off between coverage and efficiency.

---

## 📊 Dataset Selection

We utilized standard benchmarks from the **`bnlearn` repository**, widely recognized as the gold standard for Bayesian Network research.

### Included Datasets (from Medium to Massive)
* `mildew` (35 nodes), `alarm` (37 nodes), `barley` (48 nodes)
* `hepar2` (70 nodes), `win95pts` (76 nodes)
* `diabetes` (413 nodes), `link` (724 nodes)
* **`munin` (1041 nodes)** — *Primary scalability test.*

### Excluded Datasets
Datasets with anonymized labels (e.g., `pathfinder` with nodes named "F1", "F2", or `andes`) were excluded. The method relies on **explicit semantic interpretation**; therefore, "black-box" variables without linguistic context cannot be effectively vectorized.

---

## 🚀 Key Results & Interpretation

### 1. Superior Scalability
Our experiments confirm that **semantic filtering becomes more effective as the graph complexity increases**.
* **MUNIN (1041 nodes):** Achieved the highest efficiency. The distinct medical terminology allows the LLM to separate unrelated clusters (e.g., distinguishing *Nerve* signals from unrelated *Muscle* groups) with near-perfect precision.
* **Small Graphs:** Showed lower efficiency due to "semantic density" (all nodes are closely related concepts).

### 2. Defining "Success" in Causal Discovery
While a **100% Recall** is the theoretical ideal, it is often computationally unattainable for massive graphs using standard statistical methods alone.
* **Result:** We demonstrate that retaining **80-90% of True Positives** while reducing the search space by **50-80%** is a significant breakthrough.
* **Justification:** It is better to recover a highly accurate partial graph in minutes than to fail to recover a perfect graph due to computational timeouts (NP-hard problem). The remaining edges can be refined by downstream algorithms (e.g., PC-stable) much faster on the sparsified skeleton.

![plot](/data/processed/plots/reduction_recall.png)
*(Figure 1: Dependence of Reduction Efficiency on Recall (Completeness))*


![plot](/data/processed/plots/reduction_number_of_nodes.png)
*(Figure 2: Dependence of Reduction Efficiency on Graph Size at fixed Recall = 20%)*

---

## 🔮 Future Directions

This research establishes a baseline for Semantic Causal Discovery. Several promising avenues for optimization have been identified:

### A. Advanced Filtering Logic
1.  **Hybrid Thresholding:** Combining **Top-K** with a **Similarity Threshold** (e.g., keep top 5 neighbors *only if* similarity > 0.7) to eliminate weak neighbors in sparse clusters.
2.  **Dynamic K-Expansion:** If a node has strong confirmed connections at small $k$, it implies a "hub" status. For such nodes, $k$ should be dynamically expanded to capture all potential children.
3.  **Orphan & Hub Analysis:**
    * Nodes that do not appear in any Top-K lists are candidates for "Orphan Recovery" (widening the search).
    * Nodes with disproportionately high True Positives are likely "Hubs" (confounders) requiring deeper analysis.

### B. Semantic Enhancement
4.  **Ensemble Embeddings:** Generating multiple definitions (perspectives) for each node and aggregating the Top-K results to reduce hallucination noise.
5.  **Lexical Denoising:** Removing instrumental/measurement words (e.g., "rate of", "level of", "sensor") from definitions to focus purely on the core entity semantics.
6.  **Causal Strength vs. Semantic Distance:** Investigating the hypothesis that "missed" edges (False Negatives) correspond to weak causal links that have low semantic similarity (e.g., indirect side effects).

### C. Technical Improvements
7.  **Model Scaling:** Testing SOTA models (e.g., Gemini 1.5 Pro, GPT-4) to assess if "reasoning" capabilities improve embedding quality.
8.  **Alternative Metrics:** Experimenting with non-cosine similarities (e.g., Euclidean, Manhatten) or learned metric spaces.
9.  **Stochastic Sampling:** Varying embedding temperature to capture broader semantic nuances.

---

## ⚠️ Limitations

* **Data Availability:** This study was conducted on a finite set of open-source benchmark networks ([`bnlearn`](https://www.bnlearn.com/bnrepository/)). While results on `munin` and `link` are statistically significant, validating the method on proprietary industrial graphs (e.g., IT logs, Telecom networks) would further robustify the findings.
* **Context Dependence:** The method requires high-quality variable names. It is not suitable for anonymized datasets without metadata.

---

## 🔧 Technology Stack

* **Language:** Python 3.10+
* **LLM Provider:** Google Gemini API
    * Interpretation: `gemini-3-flash-preview`
    * Embeddings: `gemini-embedding-001`
* **Orchestration:** Custom Pipeline with API Key Rotation for rate limit handling.
* **Analysis:** `pandas`, `networkx`, `seaborn`.