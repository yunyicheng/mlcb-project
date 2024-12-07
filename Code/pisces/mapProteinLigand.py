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
import preprocessing
from itertools import combinations

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

adata, gene, cell_extended, cluster_details, cluster_colors = preprocessing.load_data()
adata = sc.read_h5ad((DATADIR /"Intermediate/merfish_w_ppi.h5ad"))
print(adata.var.shape)
# print(cell_extended.columns)

grn_data = pd.read_csv(INTERDATADIR / "grn_mapped.tsv", sep="\t")
print(grn_data.head())

tf_target_gene_ids = set(grn_data["TF_id"]).union(set(grn_data["Target_id"]))

# Filter adata to include only the relevant genes
filtered_genes = adata.var.index.intersection(tf_target_gene_ids)

# Check the number of matching genes
print(f"Number of matching genes in adata: {len(filtered_genes)}")

# Subset the adata object
adata = adata[:, filtered_genes].copy()

# Verify the new shape of adata
print(f"New shape of adata_grn: {adata.shape}")




nan_counts = cell_extended[["x", "y", "z"]].isnull().sum()

# Print the results
# print("Number of NaN values in each column:")
# print(nan_counts)
# print(adata.var.index[0])
common_cells = adata.obs.index.intersection(cell_extended.index)
adata = adata[common_cells].copy()
# print(f"New shape of adata: {adata.shape}")

adata_genes = set(adata.var.index)

# Filter protein-ligand interactions based on gene symbols
protein_ligand_data = protein_ligand_data[
    (protein_ligand_data["ligand_ensembl_gene_id"].isin(adata_genes)) &
    (protein_ligand_data["receptor_ensembl_gene_id"].isin(adata_genes))
]

print(f"Filtered protein-ligand interactions: {protein_ligand_data.shape}")
# print(protein_ligand_data)

# print(adata.var["gene_symbol"].head())
# print(adata.var)

#### set brain section ###
# Perform the join on 'brain_section_label'
# Rename overlapping columns in cell_extended
cell_extended = cell_extended.rename(
    columns={"brain_section_label": "brain_section_label_extended"}
)

# Join metadata with AnnData
adata.obs = adata.obs.join(cell_extended, how="left")
# adata.obs = adata.obs.dropna(subset=["x", "y", "z"]) 

# print(adata.obs.head())
# Get min and max values for each coordinate
x_min, x_max = adata.obs["x"].min(), adata.obs["x"].max()
y_min, y_max = adata.obs["y"].min(), adata.obs["y"].max()
z_min, z_max = adata.obs["z"].min(), adata.obs["z"].max()

# Define bins
padding = 1e-6  # Small padding to include edge cases
x_bins = np.linspace(x_min - padding, x_max + padding, 5)
y_bins = np.linspace(y_min - padding, y_max + padding, 5)
z_bins = np.linspace(z_min - padding, z_max + padding, 5)
# Perform binning
adata.obs["x_bin"] = pd.cut(adata.obs["x"], bins=x_bins, labels=range(4))
adata.obs["y_bin"] = pd.cut(adata.obs["y"], bins=y_bins, labels=range(4))
adata.obs["z_bin"] = pd.cut(adata.obs["z"], bins=z_bins, labels=range(4))

# Check for NaN values in bins
# print(f"x_bin NaNs: {adata.obs['x_bin'].isnull().sum()}")
# print(f"y_bin NaNs: {adata.obs['y_bin'].isnull().sum()}")
# print(f"z_bin NaNs: {adata.obs['z_bin'].isnull().sum()}")

# Convert bin labels to strings and construct the section labels
adata.obs["section"] = (
    adata.obs["x_bin"].astype(str) + "_" +
    adata.obs["y_bin"].astype(str) + "_" +
    adata.obs["z_bin"].astype(str)
)

# Verify the section labels
# print(adata.obs["section"].head())
# cell_label
# 1015221640100570404    1_2_0
# 1015221640100800173    2_2_0
# 1015221640100360300    1_2_0
# 1015221640100800090    2_2_0
# 1015221640100810318    2_2_0

# Verify section assignment
# print(adata.obs["section"].value_counts())
# print(adata.obs.head())


###### MAHATTEN THRESHOLD ######
# # Inspect the range of spatial coordinates
# print(f"x range: {adata.obs['x'].min()} to {adata.obs['x'].max()}")
# print(f"y range: {adata.obs['y'].min()} to {adata.obs['y'].max()}")
# print(f"z range: {adata.obs['z'].min()} to {adata.obs['z'].max()}")
# import numpy as np
# from itertools import combinations

# # Get spatial coordinates
# coords = adata.obs[["x", "y", "z"]].values

# # Sample 1000 random pairs of cells
# random_pairs = np.random.choice(range(len(coords)), size=(1000, 2), replace=False)
# distances = [np.sum(np.abs(coords[i] - coords[j])) for i, j in random_pairs]

# # Inspect statistics of the distances
# print(f"Manhattan distances: min={np.min(distances)}, max={np.max(distances)}, mean={np.mean(distances)}, median={np.median(distances)}")
# # Set threshold based on percentiles
# threshold_25 = np.percentile(distances, 25)  # 25th percentile
# threshold_50 = np.percentile(distances, 50)  # Median
# threshold_75 = np.percentile(distances, 75)  # 75th percentile

# print(f"Suggested thresholds: 25th percentile={threshold_25}, median={threshold_50}, 75th percentile={threshold_75}")
# x range: 0.4609526242692965 to 10.632294059799506
# y range: 1.386056075501612 to 9.274293635507194
# z range: 0.0 to 15.0
# Manhattan distances: min=0.09056093737691917, max=18.19045663947075, mean=7.8915637805290695, median=7.639213171270928
# Suggested thresholds: 25th percentile=5.447948535439315, median=7.639213171270928, 75th percentile=10.142603834729846	
    # •	25th Percentile:  5.45  – Use for selecting close interactions.
	# •	Median:  7.64  – Balances between close and broader interactions.
	# •	75th Percentile:  10.14  – Includes a broader range of interactions.


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

manhattan_threshold = 5.45  

for brain_section, indices in brain_section_groups.groups.items():
    brain_section_adata[brain_section] = adata[indices].copy()
    print(f"Created AnnData for brain section '{brain_section}' with {brain_section_adata[brain_section].n_obs} cells.")


# Dictionary to store graphs for each brain section
brain_section_graphs = {}

for brain_section in brain_section_adata:
    adata_section = brain_section_adata[brain_section]
    # adata_section = brain_section_adata['C57BL6J-638850.69'] # adata for this brain section
    section_graphs = {}
    spatial_section_groups = adata_section.obs.groupby("section")

    spatial_section_adata = {}
    for spatial_section, indices in spatial_section_groups.groups.items():
        spatial_section_adata[spatial_section] = adata_section[indices].copy()
        print(f"Created AnnData for spatial section '{spatial_section}' with {spatial_section_adata[spatial_section].n_obs} cells.")


    for section_label in spatial_section_adata:
        print(f"Processing section: {section_label}")
        adata_spatial = spatial_section_adata[section_label]
        # Initialize a graph for this brain section
        G = nx.Graph()
    
        # adata_section = brain_section_adata[brain_section] # adata for this brain section
        expression_matrix = adata_spatial.X
        # Add nodes (cells) with metadata as attributes
        cell_ids = adata_spatial.obs.index.tolist()
        adata_genes_spatial = set(adata_spatial.var.index)

        cell_pairs = list(combinations(range(len(cell_ids)), 2))
        # cell_pairs = []
        # for i, j in combinations(range(len(cell_ids)), 2):
        #     coord1 = adata_spatial.obs.loc[cell_ids[i], ["x", "y", "z"]].values
        #     coord2 = adata_spatial.obs.loc[cell_ids[j], ["x", "y", "z"]].values
        #     manhattan_distance = np.sum(np.abs(coord1 - coord2))

        #     if manhattan_distance <= manhattan_threshold:
        #         cell_pairs.append((i, j))

        # Filter protein-ligand interactions based on gene symbols
        filtered_protein_ligand_data = protein_ligand_data[
            (protein_ligand_data["ligand_ensembl_gene_id"].isin(adata_genes_spatial)) &
            (protein_ligand_data["receptor_ensembl_gene_id"].isin(adata_genes_spatial))
        ]

        for cell_idx, cell_id in enumerate(cell_ids):
            # Average gene expression for each cell
            avg_expression = np.mean(expression_matrix[cell_idx, :].toarray())  # Use `.toarray()` if sparse
            G.add_node(cell_id, average_expression=avg_expression)


        # Map gene names to column indices in the expression matrix
        gene_to_idx = {gene: idx for idx, gene in enumerate(adata_spatial.var["gene_symbol"])}
        gene_id_to_symbol = dict(zip(adata_spatial.var.index, adata_spatial.var["gene_symbol"]))

        i = 0
        # Add ligand-receptor edges
        for _, row in tqdm(filtered_protein_ligand_data.iterrows()):
            ligand_gene = gene_id_to_symbol.get(row["ligand_ensembl_gene_id"])
            receptor_gene = gene_id_to_symbol.get(row["receptor_ensembl_gene_id"])

            ligand_idx = gene_to_idx.get(ligand_gene)
            receptor_idx = gene_to_idx.get(receptor_gene)

            # Add edges for ligand-receptor coexpression
            if ligand_idx is not None and receptor_idx is not None:
                # # Extract ligand and receptor expression vectors for all cells
                # ligand_expr = expression_matrix[:, ligand_idx].toarray().flatten()
                # receptor_expr = expression_matrix[:, receptor_idx].toarray().flatten()

                # # Create a boolean mask for valid ligand-receptor coexpression
                # expr_mask = np.outer(ligand_expr > 0, receptor_expr > 0)

                # # Iterate through valid cell pairs
                # for i, j in tqdm(cell_pairs, desc="Cell Pairs"):
                #     if expr_mask[i, j]:  # Both ligand and receptor expressed
                #         coexpression_score = ligand_expr[i] * receptor_expr[j]
                #         G.add_edge(cell_ids[i], cell_ids[j], weight=coexpression_score, type="ligand-receptor")
                #     # Add spatial distance edge
                #     coord1 = adata_spatial.obs.loc[cell_ids[i], ["x", "y", "z"]].values
                #     coord2 = adata_spatial.obs.loc[cell_ids[j], ["x", "y", "z"]].values
                #     manhattan_distance = np.sum(np.abs(coord1 - coord2))
                #     G.add_edge(cell1, cell2, weight=1 / manhattan_distance, type="spatial")


                for i, j in tqdm(cell_pairs, desc="Cell Pairs"):
                    cell1, cell2 = cell_ids[i], cell_ids[j]

                    # Calculate Manhattan distance between cells
                    coord1 = adata_spatial.obs.loc[cell1, ["x", "y", "z"]].values
                    coord2 = adata_spatial.obs.loc[cell2, ["x", "y", "z"]].values
                    manhattan_distance = np.sum(np.abs(coord1 - coord2))

                    # Skip cell pairs that exceed the Manhattan distance threshold
                    if manhattan_distance > manhattan_threshold:
                        continue
                    
                    # if expr_mask[i, j]:  # Both ligand and receptor expressed
                    #     coexpression_score = ligand_expr[i] * receptor_expr[j]
                    #     t = "ligand-receptor" + row["lr_pair"]
                    #     G.add_edge(cell_ids[i], cell_ids[j], weight=coexpression_score, type=t)

                    # Add ligand-receptor coexpression edge
                    ligand_expr = expression_matrix[i, ligand_idx]
                    receptor_expr = expression_matrix[j, receptor_idx]

                    if ligand_expr > 0 and receptor_expr > 0:  # Check for non-zero expression
                        coexpression_score = ligand_expr * receptor_expr
                        t = "ligand-receptor_" + row["lr_pair"]
                        G.add_edge(cell1, cell2, weight=coexpression_score, type=t)
                        #print(f"Added edge between {cell1} and {cell2} with weight {coexpression_score} and type {t}")

                    # Add spatial distance edge
                    if i == 0:
                        manhattan_distance = np.sum(np.abs(coord1 - coord2))
                        G.add_edge(cell1, cell2, weight=1 / manhattan_distance, type="spatial")
                i += 1

        #         # for i, cell1 in tqdm(enumerate(cell_ids), desc="Cell1 loop"):
        #         #     for j, cell2 in tqdm(enumerate(cell_ids), desc="Cell2 loop"):
        #         #         if i != j:  # Exclude self-loops
        #         #             ligand_expr = expression_matrix[i, ligand_idx]
        #         #             receptor_expr = expression_matrix[j, receptor_idx]

        #         #             if ligand_expr > 0 and receptor_expr > 0:  # Check for non-zero expression
        #         #                 coexpression_score = ligand_expr * receptor_expr
        #         #                 G.add_edge(cell1, cell2, weight=coexpression_score, type="ligand-receptor")
        # brain_section = 'C57BL6J-638850.69'
        # output_path_1 = Path(GRAPHOUTDIR) / "cell_ligand_receptor_graph.gpickle"
        output_path = Path(GRAPHOUTDIR) / f"{brain_section}_{section_label}_cell_ligand_receptor_graph.gpickle"
        # os.makedirs(output_path, exist_ok=True)
        # os.makedirs(output_path_1, exist_ok=True)

        # with open(output_path_1, "wb") as f:
        #     pickle.dump(G, f) 
        with open(output_path, "wb") as f:
            pickle.dump(G, f) 
    #     break
    # break

        

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


"""
seperate adata into brain sections
seprate adata into 4 * 4 * 4 spatial section

for each brain section, for each spatail section construct a graph
Nodes:
	•	Represent cells with their average gene expression as a feature.
Edges:
	•	Ligand-receptor coexpression edges are added only if:
	•	The Manhattan distance is within the threshold.
	•	Both ligand and receptor genes are expressed in the respective cells.
	•	Spatial proximity edges are added using Manhattan distance.


Mahattan distance threshold: 7.64

x range: 0.4609526242692965 to 10.632294059799506
y range: 1.386056075501612 to 9.274293635507194
z range: 0.0 to 15.0
Manhattan distances: min=0.09056093737691917, max=18.19045663947075, mean=7.8915637805290695, median=7.639213171270928
Suggested thresholds: 25th percentile=5.447948535439315, median=7.639213171270928, 75th percentile=10.142603834729846	
    •	25th Percentile:  5.45  – Use for selecting close interactions.
	•	Median:  7.64  – Balances between close and broader interactions.
	•	75th Percentile:  10.14  – Includes a broader range of interactions


"""