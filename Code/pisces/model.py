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
from torch_geometric.nn import GATConv

from .config import *


class GATSelfSupervised(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4):
        super(GATSelfSupervised, self).__init__()

        # First GAT layer
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)

        # Second GAT layer
        self.gat2 = GATConv(
            hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True
        )

        # Combine edge embeddings and edge attributes
        self.edge_projector = nn.Linear(
            hidden_channels * num_heads * 2 + 2, hidden_channels
        )

        # Output layer for edge prediction
        self.edge_predictor = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, edge_mask=None):
        # Node embeddings
        h = self.gat1(x, edge_index)
        h = torch.relu(h)
        h = self.gat2(h, edge_index)
        h = torch.relu(h)

        # Edge embedding (concatenate node pairs)
        edge_embeddings = torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=1)

        # Incorporate edge attributes
        edge_features = torch.cat([edge_embeddings, edge_attr], dim=1)
        edge_features = torch.relu(self.edge_projector(edge_features))

        # Predict coexpression scores
        edge_predictions = self.edge_predictor(edge_features)

        # Apply mask if provided
        if edge_mask is not None:
            edge_predictions = edge_predictions[edge_mask]

        return edge_predictions


def load_graphs_from_pickle(directory):
    data_list = []
    node_features = torch.load(OUTDIR / "nodeFeature.pkl")
    for filename in os.listdir(directory):
        if filename.endswith(".gpickle"):
            file_path = os.path.join(directory, filename)

            # Load the graph from gpickle
            with open(file_path, "rb") as f:
                nx_graph = pickle.load(f)

            # Map node IDs from the graph to node feature indices
            node_ids = list(nx_graph.nodes())
            node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

            # Extract node features for the current graph
            node_features_graph = torch.stack(
                [node_features[node_id_to_index[node]] for node in node_ids], dim=0
            )

            # Extract edge indices
            edge_index = (
                torch.tensor(
                    [
                        (node_id_to_index[u], node_id_to_index[v])
                        for u, v in nx_graph.edges
                    ],
                    dtype=torch.long,
                )
                .t()
                .contiguous()
            )

            # Extract edge attributes
            coexpression_scores = []
            distances = []

            for edge in nx_graph.edges(data=True):
                edge_data = edge[2]  # Get the dictionary of edge attributes

                # Extract coexpression score
                coexpression_score = 0.0
                for key in edge_data.keys():
                    if key.startswith("ligand-receptor_"):
                        coexpression_score = edge_data[key]
                        break  # Assume only one ligand-receptor key per edge

                # Extract spatial distance
                distance = edge_data.get("spatial", 0.0)

                coexpression_scores.append(coexpression_score)
                print(coexpression_score) if coexpression_score != 0.0 else None
                distances.append(distance)

            # Convert edge attributes to tensors
            coexpression_scores = torch.tensor(coexpression_scores, dtype=torch.float)
            distances = torch.tensor(distances, dtype=torch.float)

            # Stack edge attributes into a single tensor
            edge_attr = torch.stack([coexpression_scores, distances], dim=1)

            # Create PyTorch Geometric Data object
            data = Data(
                x=node_features_graph, edge_index=edge_index, edge_attr=edge_attr
            )
            data_list.append(data)

    return data_list
