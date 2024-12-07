import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
from Code.pisces.preprocessing import load_data

def analyze_cohort():
    # Load the data
    adata, gene, cell_extended, cluster_details, cluster_colors = load_data()
    
    # Summarize the data
    print("Gene DataFrame Info:")
    print(gene.info())
    print("\nCell Metadata Info:")
    print(cell_extended.info())
    
    # Cohort Analysis 1: Cluster-level distribution
    cluster_distribution = cell_extended['cluster_alias'].value_counts()
    print("\nCluster Distribution:")
    print(cluster_distribution)

    # Plot cluster distribution
    plt.figure(figsize=(16, 10))  # Larger dimensions
    top_clusters = cluster_distribution.head(20)  # Display top 20 clusters for better readability
    sns.barplot(
        x=top_clusters.index, 
        y=top_clusters.values, 
        color="gray",
        dodge=False,
        palette="viridis"
    )
    plt.legend([], [], frameon=False)  # Remove the legend
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
    plt.title("Top 20 Clusters by Number of Cells")
    plt.xlabel("Cluster Alias")
    plt.ylabel("Number of Cells")
    plt.tight_layout()
    # Save plot using a relative path
    results_dir = Path(__file__).resolve().parent.parent.parent / "Results"  # Navigate to the Data directory
    results_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    output_path = results_dir / "cluster_distribution.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

    print(f"Cluster distribution plot saved at: {output_path}")

    # Cohort Analysis 2: Gene expression by cluster

    # Define gene_expression before calculating variability
    selected_genes = gene.sample(5, random_state=42).index  # Randomly select 5 genes
    gene_expression = pd.DataFrame(
        sparse.csr_matrix(adata[:, selected_genes].X).todense(),
        columns=selected_genes,
        index=adata.obs_names
    )
    gene_expression['cluster_alias'] = cell_extended['cluster_alias']

    # Select the single most variable gene
    variability = gene_expression.drop(columns=["cluster_alias"]).var(axis=0)
    selected_gene = variability.nlargest(1).index[0]
    print(f"\nMost variable gene: {selected_gene}")

    # Filter the data for this gene
    gene_expression_filtered = pd.DataFrame(
        sparse.csr_matrix(adata[:, [selected_gene]].X).todense(),
        columns=[selected_gene],
        index=adata.obs_names
    )
    gene_expression_filtered['cluster_alias'] = cell_extended['cluster_alias']

    # Filter data for top 5 clusters by cell count
    top_clusters = cell_extended['cluster_alias'].value_counts().index[:5]
    filtered_gene_expression = gene_expression_filtered[
        gene_expression_filtered['cluster_alias'].isin(top_clusters)
    ]

    # Melt for visualization
    filtered_gene_expression_melted = filtered_gene_expression.melt(
        id_vars="cluster_alias",
        var_name="Gene",
        value_name="Expression"
    )

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=filtered_gene_expression_melted,
        x="cluster_alias",
        y="Expression",
        palette="Set3",
        showfliers=False  # Hide outliers for clarity
    )
    plt.title(f"Expression of {selected_gene} by Top 5 Clusters", fontsize=14)
    plt.xlabel("Cluster Alias", fontsize=12)
    plt.ylabel("Expression Level", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()

    # Save to Results directory
    gene_expression_path = results_dir / f"{selected_gene}_expression_by_cluster.png"
    plt.savefig(gene_expression_path, bbox_inches="tight")
    plt.show()

    print(f"Gene expression plot saved at: {gene_expression_path}")

    # Cohort analysis 3: Spatial Visualization of Cells
    plt.figure(figsize=(10, 10))

    # Extract cell positions (ensure x and y columns exist in cell_extended)
    cell_positions = cell_extended[['x', 'y']].dropna()

    # Use the top 5 clusters to avoid excessive overlap
    top_clusters = cell_extended['cluster_alias'].value_counts().index[:5]
    filtered_positions = cell_positions[
        cell_positions.index.isin(
            cell_extended[cell_extended['cluster_alias'].isin(top_clusters)].index
        )
    ]

    sns.scatterplot(
        data=filtered_positions,
        x="x",
        y="y",
        hue=cell_extended.loc[filtered_positions.index, "cluster_alias"],
        palette="tab20",
        alpha=0.6,
        s=5,  # Smaller points to reduce density
        linewidth=0
    )
    plt.title("Spatial Distribution of Cells (Top 5 Clusters)", fontsize=14)
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(
        loc="upper left", bbox_to_anchor=(1.05, 1), title="Cluster", fontsize=8
    )
    plt.tight_layout()

    # Save to Results directory
    spatial_distribution_path = results_dir / "spatial_distribution.png"
    plt.savefig(spatial_distribution_path, bbox_inches="tight")
    plt.show()

    print(f"Spatial distribution plot saved at: {spatial_distribution_path}")

if __name__ == "__main__":
    analyze_cohort()