
from config import *
import pandas as pd
import scanpy as sc
import numpy as np
import requests
import os
from pathlib import Path
import requests
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from itertools import combinations

gpickle_file = Path(GRAPHOUTDIR) / "C57BL6J-638850.69_1_1_3_cell_ligand_receptor_graph.gpickle"
metafile = Path(DATADIR) /'metadata/MERFISH-C57BL6J-638850/20241115/cell_metadata.csv'
cell_metadata = pd.read_csv(metafile)
print('a')
# Load the graph
with open(gpickle_file, "rb") as f:
    G = pickle.load(f)


# Map cell IDs (partial match) to x, y positions
cell_positions = {}
# for node in G.nodes:
#     matching_row = cell_metadata[cell_metadata["cell_label"].str.contains(node, na=False)]
#     if not matching_row.empty:
#         cell_positions[node] = (matching_row.iloc[0]["x"], matching_row.iloc[0]["y"])
cell_positions = cell_metadata.set_index("cell_label")[["x", "y"]].to_dict("index")

print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Nodes: {list(G.nodes())[:10]}")  # Show the first 10 nodes
print(f"Edges: {list(G.edges(data=True))[:10]}")  # Show the first 10 edges with attributes

# Filter only ligand-receptor edges and group by ligand-receptor pair
ligand_receptor_edges = defaultdict(list)

for u, v, data in G.edges(data=True):
    if data["type"].startswith("ligand-receptor_"):  # Check if edge type is ligand-receptor
        pair = data["type"].split("_", 1)[1]  # Extract ligand-receptor pair name
        ligand_receptor_edges[pair].append((u, v))

# Plot the graph
plt.figure(figsize=(12, 12))

# Plot the nodes (cells) based on their x and y coordinates
node_positions = {node: (cell_positions[node]["x"], cell_positions[node]["y"]) for node in G.nodes if node in cell_positions}

nx.draw_networkx_nodes(G, pos=node_positions, node_size=50, alpha=0.8)

coords = np.array(list(node_positions.values()))

manhattan_distances = [
    (i, j, abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1]))
    for i, j in combinations(range(len(node_positions)), 2)
]

# Get the 25th percentile threshold
distances = [dist for _, _, dist in manhattan_distances]
threshold_25 = np.percentile(distances, 25)
print(f"25th percentile Manhattan distance threshold: {threshold_25}")

node_list = list(node_positions.keys())
valid_pairs = {(node_list[i], node_list[j]) for i, j, dist in manhattan_distances if dist <= threshold_25}
# valid_pairs = {(node_positions[i], node_positions[j]) for i, j, dist in manhattan_distances if dist <= threshold_25}

# Filter only ligand-receptor edges and group by ligand-receptor pair
ligand_receptor_edges = defaultdict(list)
for u, v, data in G.edges(data=True):
    if data["type"].startswith("ligand-receptor_") and (u, v) in valid_pairs:
        pair = data["type"].split("_", 1)[1]  # Extract ligand-receptor pair name
        ligand_receptor_edges[pair].append((u, v))

print(len(ligand_receptor_edges))
for pair in ligand_receptor_edges:
    print(pair, len(ligand_receptor_edges[pair]))
# Define a colormap for edges
colors = plt.cm.get_cmap("tab10", len(ligand_receptor_edges))  # Create a colormap
# colors = plt.colormaps["tab10"]
# colors = [colors(i) for i in range(len(ligand_receptor_edges))]

# Draw edges for each ligand-receptor pair, with unique colors
for idx, (pair, edges) in enumerate(ligand_receptor_edges.items()):
    nx.draw_networkx_edges(
        G,
        pos=node_positions,
        edgelist=edges,
        edge_color=[colors(idx)],
        label=pair,
        alpha=0.7,
        width=1.5
    )

# Add a legend for ligand-receptor pairs
plt.legend(
    handles=[
        plt.Line2D([0], [0], color=colors(i), lw=2, label=pair) for i, pair in enumerate(ligand_receptor_edges.keys())
    ],
    loc="upper right",
    title="Ligand-Receptor Pairs"
)
output_plot_path = "ligand_receptor_interactions.png" 
plt.title("Graph for brain section C57BL6J-638850.69 with spatial section 1_1_3")
plt.axis("off")
# Save the plot instead of showing it
plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")  # Save with high resolution
plt.close()  # Close the plot to free memory

# Create separate plots for each ligand-receptor pair
for idx, (pair, edges) in enumerate(ligand_receptor_edges.items()):
    plt.figure(figsize=(12, 12))

    # Plot the nodes (cells) based on their x and y coordinates
    nx.draw_networkx_nodes(
        G,
        pos=node_positions,
        node_size=50,
        alpha=0.8
    )

    # Plot the edges for the current ligand-receptor pair
    nx.draw_networkx_edges(
        G,
        pos=node_positions,
        edgelist=edges,
        edge_color="r",  # Use a fixed color for this pair
        label=pair,
        alpha=0.7,
        width=1.5
    )

    # Add a title and legend for the current plot
    plt.title(f"Graph for brain section C57BL6J-638850.69 with spatial section 1_1_3 with Ligand-Receptor Interactions: {pair}")
    plt.axis("off")

    # Save the plot
    output_plot_path = f"{pair}_interactions.png"
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # print(f"Plot saved for ligand-receptor pair: {pair}")

# # Print node attributes for a specific node
# for node, attrs in G.nodes(data=True):
#     print(f"Node {node}: {attrs}")
#     break  # Print for the first node only

# # Print edge attributes for a specific edge
# for u, v, attrs in G.edges(data=True):
#     print(f"Edge ({u}, {v}): {attrs}")
#     break  # Print for the first edge only

# # Draw the graph
# plt.figure(figsize=(10, 7))
# nx.draw(G, node_size=50, node_color="lightblue", font_size=10)
# output_path = "graph_visualization.png"
# plt.savefig(output_path, format="png", dpi=300)
# plt.close()  # Close the figure to release memory