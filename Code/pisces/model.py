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
