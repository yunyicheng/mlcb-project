
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

gpickle_file = Path(DATADIR) / "cell_ligand_receptor_graph.gpickle"

# Load the graph
with open(gpickle_file, "rb") as f:
    G = pickle.load(f)

print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")