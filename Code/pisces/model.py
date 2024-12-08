import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data


def load_graphs_from_pickle(directory):
    data_list = []

    for filename in os.listdir(directory):
        if filename.endswith(".gpikle"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "rb") as f:
                # Load the graph from pickle
                nx_graph = pickle.load(f)

                # Extract node features
                node_features = torch.tensor(
                    [nx_graph.nodes[node]["feature"] for node in nx_graph.nodes()],
                    dtype=torch.float,
                )

                # Extract edge indices
                edge_index = (
                    torch.tensor(list(nx_graph.edges), dtype=torch.long)
                    .t()
                    .contiguous()
                )

                # Extract edge weights (optional)
                edge_weights = torch.tensor(
                    [
                        nx_graph.edges[edge].get("weight", 1.0)
                        for edge in nx_graph.edges()
                    ],
                    dtype=torch.float,
                )

                # Create PyTorch Geometric Data object
                data = Data(
                    x=node_features, edge_index=edge_index, edge_attr=edge_weights
                )
                data_list.append(data)

    return data_list
