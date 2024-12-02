import os
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
from scipy import sparse

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
    cluster_details.set_index("cluster_alias", inplace=True)
    cluster_colors = abc_cache.get_metadata_dataframe(
        directory="WMB-taxonomy",
        file_name="cluster_to_cluster_annotation_membership_color",
    )
    cluster_colors.set_index("cluster_alias", inplace=True)
    cell_extended = cell.join(cluster_details, on="cluster_alias")
    cell_extended = cell_extended.join(cluster_colors, on="cluster_alias")

    file = abc_cache.get_data_path(
        directory="MERFISH-C57BL6J-638850", file_name="C57BL6J-638850/log2"
    )
    adata = anndata.read_h5ad(file, backed="r")

    return adata, gene, cell_extended, cluster_details, cluster_colors
