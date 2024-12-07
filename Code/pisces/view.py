
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

gpickle_file = Path(GRAPHOUTDIR) / "C57BL6J-638850.69_1_1_3_cell_ligand_receptor_graph.gpickle"

# Load the graph
with open(gpickle_file, "rb") as f:
    G = pickle.load(f)

print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Nodes: {list(G.nodes())[:10]}")  # Show the first 10 nodes
print(f"Edges: {list(G.edges(data=True))[:10]}")  # Show the first 10 edges with attributes

# Print node attributes for a specific node
for node, attrs in G.nodes(data=True):
    print(f"Node {node}: {attrs}")
    break  # Print for the first node only

# Print edge attributes for a specific edge
for u, v, attrs in G.edges(data=True):
    print(f"Edge ({u}, {v}): {attrs}")
    break  # Print for the first edge only

# Draw the graph
plt.figure(figsize=(10, 7))
nx.draw(G, node_size=50, node_color="lightblue", font_size=10)
output_path = "graph_visualization.png"
plt.savefig(output_path, format="png", dpi=300)
plt.close()  # Close the figure to release memory