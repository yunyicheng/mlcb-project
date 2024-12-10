import os
import pickle
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from .config import *


def print_column_info(df):
    for c in df.columns:
        grouped = df[[c]].groupby(c).count()
        members = ""
        if len(grouped) < 30:
            members = str(list(grouped.index))
        print("Number of unique %s = %d %s" % (c, len(grouped), members))


def plot_section(xx, yy, cc=None, val=None, fig_width=8, fig_height=8, cmap=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_width, fig_height)
    if cmap is not None:
        plt.scatter(xx, yy, s=0.5, c=val, marker=".", cmap=cmap)
    elif cc is not None:
        plt.scatter(xx, yy, s=0.5, color=cc, marker=".")
    ax.set_ylim(11, 0)
    ax.set_xlim(0, 11)
    ax.axis("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def load_data():
    print("Loading MERFISH data")
    download_base = Path(DATADIR)
    abc_cache = AbcProjectCache.from_cache_dir(download_base)
    cell = abc_cache.get_metadata_dataframe(
        directory="MERFISH-C57BL6J-638850",
        file_name="cell_metadata",
        dtype={"cell_label": str},
    )
    cell.set_index("cell_label", inplace=True)

    gene = abc_cache.get_metadata_dataframe(
        directory="MERFISH-C57BL6J-638850", file_name="gene"
    )
    gene.set_index("gene_identifier", inplace=True)

    cluster_details = abc_cache.get_metadata_dataframe(
        directory="WMB-taxonomy",
        file_name="cluster_to_cluster_annotation_membership_pivoted",
        keep_default_na=False,
    )
    print("Here")
    cluster_details.set_index("cluster_alias", inplace=True)
    cluster_colors = abc_cache.get_metadata_dataframe(
        directory="WMB-taxonomy",
        file_name="cluster_to_cluster_annotation_membership_color",
    )
    cluster_colors.set_index("cluster_alias", inplace=True)
    cell_extended = cell.join(cluster_details, on="cluster_alias")
    cell_extended = cell_extended.join(cluster_colors, on="cluster_alias")
    print("Here")
    file = abc_cache.get_data_path(
        directory="MERFISH-C57BL6J-638850", file_name="C57BL6J-638850/log2"
    )
    print("Here")
    adata = anndata.read_h5ad(file, backed="r")
    print("MERFISH data loaded successfully")

    return adata, gene, cell_extended, cluster_details, cluster_colors


def calculate_node_features(n_top_genes=500):
    # Identify highly variable genes (if not already selected)
    adata = adata = sc.read_h5ad(
        DATADIR
        / "expression_matrices/MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-log2.h5ad"
    )
    if "highly_variable" not in adata.var:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        # Save the data
        adata.write(DATADIR / "expression_matrices/merfish_processed.h5ad")

    # Subset the data to highly variable genes
    adata = adata[:, adata.var.highly_variable]

    # Extract the expression matrix for node features
    node_features = adata.X
    # Convert to a dense tensor if sparse
    if isinstance(node_features, np.ndarray):
        node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    else:  # If sparse, convert to dense first
        node_features_tensor = torch.tensor(node_features.toarray(), dtype=torch.float)

    # Save to pickle
    with open(OUTDIR / "nodeFeature.pkl", "wb") as f:
        torch.save(node_features_tensor, f)

    return node_features_tensor


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


def summarize_train_data():
    train_data = load_graphs_from_pickle(GRAPHOUTDIR)
    summary = {
        "num_graphs": len(train_data),
        "num_nodes_per_graph": [],
        "num_edges_per_graph": [],
        "num_ligand_receptor_edges_per_graph": [],
        "num_spatial_edges_per_graph": [],
        "node_feature_dims": [],
        "ligand_edge_attr_dims": [],
        "spatial_edge_attr_dims": [],
    }

    for data in train_data:
        # Number of nodes
        summary["num_nodes_per_graph"].append(
            data.x.size(0) if data.x is not None else 0
        )

        # Number of edges
        if (
            data.edge_index is not None
            and data.edge_index.dim() == 2
            and data.edge_index.size(0) == 2
        ):
            summary["num_edges_per_graph"].append(data.edge_index.size(1))
        else:
            print(f"Warning: Missing or malformed edge_index in graph: {data}")
            summary["num_edges_per_graph"].append(0)

        # Node feature dimensions
        summary["node_feature_dims"].append(data.x.size(1) if data.x is not None else 0)

        # Ligand-receptor edge attributes
        if data.edge_attr is not None:
            summary["ligand_edge_attr_dims"].append(data.edge_attr.size(1))
        else:
            summary["ligand_edge_attr_dims"].append(0)

        # Spatial edges
        if hasattr(data, "spatial_edge_index") and data.spatial_edge_index is not None:
            if (
                data.spatial_edge_index.dim() == 2
                and data.spatial_edge_index.size(0) == 2
            ):
                summary["num_spatial_edges_per_graph"].append(
                    data.spatial_edge_index.size(1)
                )
            else:
                print(f"Warning: Malformed spatial_edge_index in graph: {data}")
                summary["num_spatial_edges_per_graph"].append(0)
        else:
            summary["num_spatial_edges_per_graph"].append(0)

        if hasattr(data, "spatial_edge_attr") and data.spatial_edge_attr is not None:
            summary["spatial_edge_attr_dims"].append(data.spatial_edge_attr.size(1))
        else:
            summary["spatial_edge_attr_dims"].append(0)

        # Ligand-receptor edges
        if (
            data.edge_attr is not None
            and data.edge_index is not None
            and data.edge_index.size(0) == 2
        ):
            num_ligand_edges = data.edge_index.size(1)
            summary["num_ligand_receptor_edges_per_graph"].append(num_ligand_edges)
        else:
            summary["num_ligand_receptor_edges_per_graph"].append(0)

    # Compute aggregate statistics
    summary["num_nodes_per_graph"] = {
        "mean": np.mean(summary["num_nodes_per_graph"]),
        "std": np.std(summary["num_nodes_per_graph"]),
        "min": np.min(summary["num_nodes_per_graph"]),
        "max": np.max(summary["num_nodes_per_graph"]),
    }
    summary["num_edges_per_graph"] = {
        "mean": np.mean(summary["num_edges_per_graph"]),
        "std": np.std(summary["num_edges_per_graph"]),
        "min": np.min(summary["num_edges_per_graph"]),
        "max": np.max(summary["num_edges_per_graph"]),
    }
    summary["node_feature_dims"] = list(
        set(summary["node_feature_dims"])
    )  # Unique dimensions
    summary["ligand_edge_attr_dims"] = list(
        set(summary["ligand_edge_attr_dims"])
    )  # Unique dimensions
    summary["spatial_edge_attr_dims"] = list(
        set(summary["spatial_edge_attr_dims"])
    )  # Unique dimensions

    return summary
