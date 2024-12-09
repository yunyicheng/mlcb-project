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


class GATWithSpatialEdges(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4):
        super(GATWithSpatialEdges, self).__init__()

        # Node embedding layers
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.gat2 = GATConv(
            hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True
        )

        # Edge embedding projector for ligand-receptor edges
        self.ligand_edge_projector = nn.Linear(
            hidden_channels * num_heads * 2 + 1, hidden_channels
        )

        # Spatial edge embedding projector
        self.spatial_edge_projector = nn.Linear(
            hidden_channels * num_heads * 2 + 1, hidden_channels
        )

        # Output layer for predicting coexpression scores
        self.edge_predictor = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x,
        ligand_edge_index,
        ligand_edge_attr,
        spatial_edge_index=None,
        spatial_edge_attr=None,
    ):
        # Node embeddings
        h = self.gat1(x, ligand_edge_index)
        h = torch.relu(h)
        h = self.gat2(h, ligand_edge_index)
        h = torch.relu(h)

        # Ligand-receptor edge embeddings
        ligand_edge_embeddings = torch.cat(
            [h[ligand_edge_index[0]], h[ligand_edge_index[1]]], dim=1
        )
        ligand_features = torch.cat([ligand_edge_embeddings, ligand_edge_attr], dim=1)
        ligand_features = torch.relu(self.ligand_edge_projector(ligand_features))

        if spatial_edge_index is not None and spatial_edge_attr is not None:
            spatial_edge_embeddings = torch.cat(
                [h[spatial_edge_index[0]], h[spatial_edge_index[1]]], dim=1
            )
            spatial_features = torch.cat(
                [spatial_edge_embeddings, spatial_edge_attr], dim=1
            )
            spatial_features = torch.relu(self.spatial_edge_projector(spatial_features))

        # Predict coexpression scores (use ligand features only)
        edge_predictions = self.edge_predictor(ligand_features)

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

            # Separate edges by type
            ligand_edge_index = []
            ligand_edge_attr = []
            spatial_edge_index = []
            spatial_edge_attr = []

            for u, v, edge_data in nx_graph.edges(data=True):
                # Map node IDs to feature indices
                u_idx = node_id_to_index[u]
                v_idx = node_id_to_index[v]

                if edge_data["type"].startswith("ligand-receptor_"):
                    # Ligand-receptor edge
                    ligand_edge_index.append((u_idx, v_idx))
                    ligand_edge_attr.append(edge_data["weight"])  # Coexpression score

                elif edge_data["type"] == "spatial":
                    # Spatial edge
                    spatial_edge_index.append((u_idx, v_idx))
                    spatial_edge_attr.append(edge_data["weight"])  # Distance

            # Convert to PyTorch tensors
            ligand_edge_index = (
                torch.tensor(ligand_edge_index, dtype=torch.long).t().contiguous()
            )
            ligand_edge_attr = torch.tensor(ligand_edge_attr, dtype=torch.float)

            spatial_edge_index = (
                torch.tensor(spatial_edge_index, dtype=torch.long).t().contiguous()
            )
            spatial_edge_attr = torch.tensor(spatial_edge_attr, dtype=torch.float)

            # Create PyTorch Geometric Data object
            data = Data(
                x=node_features_graph,
                edge_index=ligand_edge_index,
                edge_attr=ligand_edge_attr.unsqueeze(1),  # Coexpression scores
                spatial_edge_index=spatial_edge_index,
                spatial_edge_attr=spatial_edge_attr.unsqueeze(1),  # Distances
            )
            data_list.append(data)

    return data_list
