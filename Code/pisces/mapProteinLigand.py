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
from tqdm import tqdm
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor

from config import *

import requests
import os

# Function to process cell pairs in parallel
def process_cell_pairs(ligand_idx, receptor_idx, cell_pairs):
    edges = []
    ligand_expr = expression_matrix[:, ligand_idx].toarray().flatten()
    receptor_expr = expression_matrix[:, receptor_idx].toarray().flatten()

    for cell1, cell2 in cell_pairs:
        if ligand_expr[cell1] > 0 and receptor_expr[cell2] > 0:
            coexpression_score = ligand_expr[cell1] * receptor_expr[cell2]
            edges.append((cell_ids[cell1], cell_ids[cell2], coexpression_score))
    return edges


# CellTalkDB
binding_data_file = Path(HOMEDIR) / "UploadData/mouse_lr_pair.txt"
protein_ligand_data = pd.read_csv(binding_data_file, sep="\t")

adata = sc.read_h5ad((DATADIR /"Intermediate/merfish_w_ppi.h5ad"))
# print(adata.var.index[0])
adata_genes = set(adata.var.index)

# Filter protein-ligand interactions based on gene symbols
protein_ligand_data = protein_ligand_data[
    (protein_ligand_data["ligand_ensembl_gene_id"].isin(adata_genes)) &
    (protein_ligand_data["receptor_ensembl_gene_id"].isin(adata_genes))
]

print(f"Filtered protein-ligand interactions: {protein_ligand_data.shape}")
#print(protein_ligand_data)

# print(adata.var["gene_symbol"].head())
# print(adata.var)


cell_metadata_file = DATADIR / "metadata/MERFISH-C57BL6J-638850/20241115/cell_metadata.csv"
cell_metadata = pd.read_csv(cell_metadata_file)
# Group cells by brain section
brain_sections = cell_metadata.groupby("brain_section_label")
# List all brain sections
# print("Brain sections:", brain_sections.groups.keys())
# Brain sections: dict_keys(['C57BL6J-638850.01', 'C57BL6J-638850.02', 'C57BL6J-638850.03', 'C57BL6J-638850.04', 'C57BL6J-638850.05', 'C57BL6J-638850.06', 'C57BL6J-638850.08', 'C57BL6J-638850.09', 'C57BL6J-638850.10', 'C57BL6J-638850.11', 'C57BL6J-638850.12', 'C57BL6J-638850.13', 'C57BL6J-638850.14', 'C57BL6J-638850.15', 'C57BL6J-638850.16', 'C57BL6J-638850.17', 'C57BL6J-638850.18', 'C57BL6J-638850.19', 'C57BL6J-638850.24', 'C57BL6J-638850.25', 'C57BL6J-638850.26', 'C57BL6J-638850.27', 'C57BL6J-638850.28', 'C57BL6J-638850.29', 'C57BL6J-638850.30', 'C57BL6J-638850.31', 'C57BL6J-638850.32', 'C57BL6J-638850.33', 'C57BL6J-638850.35', 'C57BL6J-638850.36', 'C57BL6J-638850.37', 'C57BL6J-638850.38', 'C57BL6J-638850.39', 'C57BL6J-638850.40', 'C57BL6J-638850.42', 'C57BL6J-638850.43', 'C57BL6J-638850.44', 'C57BL6J-638850.45', 'C57BL6J-638850.46', 'C57BL6J-638850.47', 'C57BL6J-638850.48', 'C57BL6J-638850.49', 'C57BL6J-638850.50', 'C57BL6J-638850.51', 'C57BL6J-638850.52', 'C57BL6J-638850.54', 'C57BL6J-638850.55', 'C57BL6J-638850.56', 'C57BL6J-638850.57', 'C57BL6J-638850.58', 'C57BL6J-638850.59', 'C57BL6J-638850.60', 'C57BL6J-638850.61', 'C57BL6J-638850.62', 'C57BL6J-638850.64', 'C57BL6J-638850.66', 'C57BL6J-638850.67', 'C57BL6J-638850.68', 'C57BL6J-638850.69'])
# Group by brain_section_label
brain_section_groups = adata.obs.groupby("brain_section_label")

# Example: Iterate through each brain section
# Create a dictionary to store AnnData for each brain section
brain_section_adata = {}

for brain_section, indices in brain_section_groups.groups.items():
    brain_section_adata[brain_section] = adata[indices].copy()
    print(f"Created AnnData for brain section '{brain_section}' with {brain_section_adata[brain_section].n_obs} cells.")


# Dictionary to store graphs for each brain section
brain_section_graphs = {}

for brain_section in brain_section_adata:
    # Initialize a graph for this brain section
    G = nx.Graph()
    
    # adata_section = brain_section_adata[brain_section] # adata for this brain section
    adata_section = brain_section_adata['C57BL6J-638850.69'] # adata for this brain section
    expression_matrix = adata_section.X
    # Add nodes (cells) with metadata as attributes
    cell_ids = adata_section.obs.index.tolist()
    adata_genes_section = set(adata_section.var.index)

    cell_pairs = list(combinations(range(len(cell_ids)), 2))

    # Filter protein-ligand interactions based on gene symbols
    filtered_protein_ligand_data = protein_ligand_data[
        (protein_ligand_data["ligand_ensembl_gene_id"].isin(adata_genes_section)) &
        (protein_ligand_data["receptor_ensembl_gene_id"].isin(adata_genes_section))
    ]

    for cell_idx, cell_id in enumerate(cell_ids):
        # Average gene expression for each cell
        avg_expression = np.mean(expression_matrix[cell_idx, :].toarray())  # Use `.toarray()` if sparse
        G.add_node(cell_id, average_expression=avg_expression)


    # Map gene names to column indices in the expression matrix
    gene_to_idx = {gene: idx for idx, gene in enumerate(adata_section.var["gene_symbol"])}
    gene_id_to_symbol = dict(zip(adata_section.var.index, adata_section.var["gene_symbol"]))

    # # Process each ligand-receptor pair
    # all_edges = []
    # with ProcessPoolExecutor() as executor:
    #     futures = []
    #     for _, row in tqdm(filtered_protein_ligand_data.iterrows(), desc="Processing ligand-receptor pairs"):
    #         ligand_gene = gene_id_to_symbol.get(row["ligand_ensembl_gene_id"])
    #         receptor_gene = gene_id_to_symbol.get(row["receptor_ensembl_gene_id"])

    #         ligand_idx = gene_to_idx.get(ligand_gene)
    #         receptor_idx = gene_to_idx.get(receptor_gene)

    #         if ligand_idx is not None and receptor_idx is not None:
    #             futures.append(executor.submit(process_cell_pairs, ligand_idx, receptor_idx, cell_pairs))

    #     for future in tqdm(futures, desc="Collecting results"):
    #         all_edges.extend(future.result())

    # # Add edges to the graph
    # for cell1, cell2, weight in all_edges:
    #     G.add_edge(cell1, cell2, weight=weight, type="ligand-receptor")

    # Add ligand-receptor edges
    for _, row in tqdm(filtered_protein_ligand_data.iterrows()):
        ligand_gene = gene_id_to_symbol.get(row["ligand_ensembl_gene_id"])
        receptor_gene = gene_id_to_symbol.get(row["receptor_ensembl_gene_id"])

        ligand_idx = gene_to_idx.get(ligand_gene)
        receptor_idx = gene_to_idx.get(receptor_gene)

        # Add edges for ligand-receptor coexpression
        if ligand_idx is not None and receptor_idx is not None:
            for i, cell1 in tqdm(enumerate(cell_ids), desc="Cell1 loop"):
                for j, cell2 in tqdm(enumerate(cell_ids), desc="Cell2 loop"):
                    if i != j:  # Exclude self-loops
                        ligand_expr = expression_matrix[i, ligand_idx]
                        receptor_expr = expression_matrix[j, receptor_idx]

                        if ligand_expr > 0 and receptor_expr > 0:  # Check for non-zero expression
                            coexpression_score = ligand_expr * receptor_expr
                            G.add_edge(cell1, cell2, weight=coexpression_score, type="ligand-receptor")

        output_path = Path(DATADIR) / "cell_ligand_receptor_graph.gpickle"
        with open(output_path, "wb") as f:
            pickle.dump(G, f)
        
        break

# print(f"Graph saved to {output_path}")

# # Define spatial proximity threshold (adjust as needed)
# spatial_threshold = 50  # Example: 50 units

# # Create graphs for each brain section
# for brain_section, group in brain_sections:
#     # Initialize a graph for this brain section
#     G = nx.Graph()
    
#     # Add nodes (cells) with metadata as attributes
#     for _, row in group.iterrows():
#         G.add_node(
#             row["cell_label"],
#             brain_section=brain_section,
#             cluster_alias=row["cluster_alias"],
#             correlation_score=row["average_correlation_score"],
#             donor_label=row["donor_label"],
#             donor_genotype=row["donor_genotype"],
#             donor_sex=row["donor_sex"],
#             position=(row["x"], row["y"], row["z"])
#         )
    
#     # Add edges based on spatial proximity
#     cell_positions = group[["cell_label", "x", "y", "z"]].set_index("cell_label").to_dict("index")
#     cell_labels = list(cell_positions.keys())
    
#     for i, cell1 in enumerate(cell_labels):
#         for j, cell2 in enumerate(cell_labels):
#             if i != j:
#                 pos1 = np.array([cell_positions[cell1]["x"], cell_positions[cell1]["y"], cell_positions[cell1]["z"]])
#                 pos2 = np.array([cell_positions[cell2]["x"], cell_positions[cell2]["y"], cell_positions[cell2]["z"]])
                
#                 distance = np.linalg.norm(pos1 - pos2)
#                 if distance < spatial_threshold:
#                     G.add_edge(cell1, cell2, weight=1 / distance)
    
#     # Store the graph
#     brain_section_graphs[brain_section] = G

#     print(f"Graph for brain section '{brain_section}' has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")




# # Map ENSMUSG to gene symbols in AnnData
# gene_to_idx = {gene: idx for idx, gene in enumerate(adata.var["gene_symbol"])}
# gene_id_to_symbol = dict(zip(adata.var.index, adata.var["gene_symbol"]))

# # print(gene_to_idx[:5])
# print('##################')
# print(gene_id_to_symbol)

# G = nx.Graph()

# # Add ligand-receptor coexpression edges
# for _, row in tqdm(protein_ligand_data.iterrows()):
#     ligand_gene = gene_id_to_symbol.get(row["ligand_ensembl_gene_id"])
#     print("ligand_ensembl_gene_id: " + str(ligand_gene))
#     receptor_gene = gene_id_to_symbol.get(row["receptor_ensembl_gene_id"])
#     print("receptor_ensembl_gene_id: " + str(receptor_gene))


#     ligand_idx = gene_to_idx.get(ligand_gene)
#     receptor_idx = gene_to_idx.get(receptor_gene)

#     print("ligand_idx: " + str(ligand_idx))
#     print("receptor_idx: " + str(receptor_idx))

#     if ligand_idx is not None and receptor_idx is not None:
#         for cell1 in tqdm(range(adata.shape[0]), desc = 'cell1 loop'):
#             for cell2 in tqdm(range(adata.shape[0]), desc = 'cell2 loop'):
#                 if cell1 != cell2:  # Exclude self-loops
#                     ligand_expr = adata.X[cell1, ligand_idx]
#                     receptor_expr = adata.X[cell2, receptor_idx]

#                     if ligand_expr > 0 and receptor_expr > 0:
#                         G.add_edge(
#                             adata.obs.index[cell1],
#                             adata.obs.index[cell2],
#                             weight=ligand_expr * receptor_expr,
#                             type="ligand-receptor"
#                         )

# if "spatial" in adata.obsm:
#     spatial_coords = adata.obsm["spatial"]

#     for i, cell1 in enumerate(adata.obs.index):
#         for j, cell2 in enumerate(adata.obs.index):
#             if i != j:  # Exclude self-loops
#                 distance = np.linalg.norm(spatial_coords[i] - spatial_coords[j])
#                 if distance < 50:  # Threshold for proximity
#                     G.add_edge(cell1, cell2, weight=1 / distance, type="spatial")

# # # Save the graph for downstream analysis
# # nx.write_gpickle(G, Path(DATADIR) / "cell_ligand_receptor_graph.gpickle")      


# # Save the graph to a file
# output_path = Path(DATADIR) / "cell_ligand_receptor_graph.gpickle"
# with open(output_path, "wb") as f:
#     pickle.dump(G, f)

# print(f"Graph saved to {output_path}")



# ppi_data = pd.DataFrame(adata.uns["string_interactions"])  # PPI interactions
# string_protein_map = adata.var["string_protein_id"]  # STRING protein mapping

# # Extract relevant STRING protein IDs from AnnData
# relevant_proteins = set(adata.var["string_protein_id"].dropna())
# print(relevant_proteins)
# print(protein_ligand_data["ligand_ensembl_protein_id"].head())

# # Filter protein-ligand interactions for relevant proteins
# protein_ligand_data = protein_ligand_data[
#     (protein_ligand_data["ligand_ensembl_protein_id"].isin(relevant_proteins)) &
#     (protein_ligand_data["receptor_ensembl_protein_id"].isin(relevant_proteins))
# ]

# print(f"Filtered protein-ligand interactions: {protein_ligand_data.shape}")


# TODO: find protein-ligand interaction db, integration ppi with protein-ligand interaction data, construct a graph
### Protein-Ligand Interaction Data from bindingDB ###

# def download_data():
#     url = "https://www.bindingdb.org/bind/BDB-Oracle_All_202412_dmp.zip"
#     output_file = Path(DATADIR)/"bindingDB/BDB-Oracle_All_202412_dmp.zip"
#     os.makedirs(output_file.parent, exist_ok=True)

#     print(f"Downloading BindingDB full dataset from {url}...")
#     response = requests.get(url, stream=True)

#     if response.status_code == 200:
#         with open(output_file, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         print(f"Downloaded to {output_file}")
#     else:
#         print(f"Failed to download BindingDB dataset. Status code: {response.status_code}")

# def unzip_data():
#     # Define the path to the zip file and the extraction directory

#     zip_file = Path(DATADIR)/"bindingDB/BDB-Oracle_All_202412_dmp.zip"
#     extraction_dir = Path(DATADIR)/"bindingDB/"
#     # Ensure the extraction directory exists
#     # Extract the zip file
#     with zipfile.ZipFile(zip_file, "r") as zip_ref:
#         zip_ref.extractall(extraction_dir)

#     print(f"Extracted {zip_file} to {extraction_dir}")


# if not Path(DATADIR / "bindingDB").exists():
#     download_data()
#     unzip_data()


# # Extract PPI data from AnnData

# adata = sc.read_h5ad((DATADIR /"Intermediate/merfish_w_ppi.h5ad"))

# ppi_data = pd.DataFrame(adata.uns["string_interactions"])  # PPI interactions
# string_protein_map = adata.var["string_protein_id"]  # STRING protein mapping
# print(ppi_data.head())


