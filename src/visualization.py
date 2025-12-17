import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from typing import Set, Tuple
import pandas as pd
import numpy as np

class Visualizer:
    """
    Helper class for generating plots and heatmaps.
    """

    @staticmethod
    def plot_error_analysis(
        sim_matrix: pd.DataFrame, 
        ground_truth_edges: Set[Tuple[str, str]], 
        predicted_edges: Set[Tuple[str, str]], 
        title: str = "Error Analysis"
    ) -> plt.Figure:
        """
        Generates a heatmap highlighting Found (Green) and Missed (Red) edges.
        
        Returns:
            plt.Figure: The matplotlib figure object (to be saved later).
        """

        # Calculate sets
        found_edges = ground_truth_edges.intersection(predicted_edges)
        missed_edges = ground_truth_edges - predicted_edges
        
        # Initialize Figure (Use a larger size for clarity)
        num_nodes = sim_matrix.shape[0]
        size = max(10, num_nodes * 0.5)
        figure, ax = plt.subplots(figsize=(size, size))
        
        # Draw Heatmap
        sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        
        name_to_idx = {name: i for i, name in enumerate(sim_matrix.columns)}

        # Highlight Found Edges (Green Solid)
        for u, v in found_edges:
            if u in name_to_idx and v in name_to_idx:
                rect = Rectangle(
                    (name_to_idx[v], name_to_idx[u]), 1, 1, 
                    fill=False, edgecolor='#00FF00', lw=3, linestyle='solid'
                )
                ax.add_patch(rect)

        # Highlight Missed Edges (Red Dashed)
        for u, v in missed_edges:
            if u in name_to_idx and v in name_to_idx:
                rect = Rectangle(
                    (name_to_idx[v], name_to_idx[u]), 1, 1, 
                    fill=False, edgecolor='#FF0000', lw=3, linestyle='--'
                )
                ax.add_patch(rect)

        # Add Legend
        legend_elements = [
            Line2D([0], [0], color='#00FF00', lw=3, label=f'Found ({len(found_edges)})'),
            Line2D([0], [0], color='#FF0000', lw=3, linestyle='--', label=f'Missed ({len(missed_edges)})')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
        
        ax.set_title(title)
        
        # Return figure object instead of just showing it
        plt.tight_layout()
        return figure

    @staticmethod
    def plot_k_sensitivity(
        df_results: pd.DataFrame, 
        dataset_name: str
    ) -> plt.Figure:
        """
        Plots the trade-off between Recall and Reduction Rate for different k values.
        
        Args:
            df_results: DataFrame with columns ['k', 'recall', 'reduction_rate']
            dataset_name: Name of the dataset for the title.
            
        Returns:
            plt.Figure: The figure object.
        """
        # Налаштування стилю
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Малюємо Recall (бажано, щоб був високим)
        sns.lineplot(
            data=df_results, x='k', y='recall', 
            ax=ax, marker='o', label='Recall (Direct)', 
            color='blue', linewidth=2.5
        )

        # Малюємо Reduction Rate (бажано, щоб був високим)
        sns.lineplot(
            data=df_results, x='k', y='reduction_rate', 
            ax=ax, marker='s', label='Reduction Rate', 
            color='red', linewidth=2.5
        )

        # Оформлення
        ax.set_title(f"Sensitivity Analysis: Top-K Strategy ({dataset_name})", fontsize=14)
        ax.set_xlabel("k (Number of Neighbors)", fontsize=12)
        ax.set_ylabel("Score (0.0 - 1.0)", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.arange(0, 1.05, 0.05))
        ax.set_xticks(df_results['k'].unique()) # Щоб на осі X були цілі числа k
        
        # Додаємо легенду
        ax.legend(fontsize=11)
        
        return fig