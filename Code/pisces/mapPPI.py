import pandas as pd
import scanpy as sc
import requests
import os
from pathlib import Path
import requests
import gzip
import shutil
import networkx as nx
import matplotlib.pyplot as plt

from config import *

import requests
import os

def download_string_data():
    base_url = "https://stringdb-downloads.org/download"
    files_to_download = [
        "protein.links.full.v12.0/10090.protein.links.full.v12.0.txt.gz",
        "protein.info.v12.0/10090.protein.info.v12.0.txt.gz",
        "protein.aliases.v12.0/10090.protein.aliases.v12.0.txt.gz"
    ]
    download_dir = Path(DATADIR) / "STRING"
    os.makedirs(download_dir, exist_ok=True)
    
    # Download each file
    for file_name in files_to_download:
        file_url = base_url + "/" + file_name
        local_subdir = download_dir / "/".join(file_name.split("/")[:-1])
        os.makedirs(local_subdir, exist_ok=True) 

        local_path = download_dir / file_name
        
        print("Downloading {}...".format(file_name))
        response = requests.get(file_url, stream=True)
        
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Saved to {}".format(local_path))
        else:
            print("Failed to download {}. Status code: {}".format(file_name, response.status_code))
    
    for f in os.listdir(download_dir):
        for file_name in os.listdir(download_dir / f):
            if file_name.endswith(".gz"):
                file_path = os.path.join(download_dir, f, file_name)
                extracted_path = os.path.join(download_dir, file_name).replace(".gz", "")
                
                print("Extracting {}...".format(file_name))
                with gzip.open(file_path, "rb") as f_in:
                    with open(extracted_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print("Extracted to {}".format(extracted_path))


# load protein-protein interaction information into adata
def load_protein_protein_interaction_data(adata, interaction_file, info_file, aliases_file):
    # Load interaction data
    interactions = pd.read_csv(interaction_file, sep=" ")
    # print(interactions.head())  # Columns: protein1, protein2, combined_score

    # Load protein information
    protein_info = pd.read_csv(info_file, sep="\t")
    # print(protein_info.head())  # Columns: protein_id, preferred_name, description

    # Load aliases
    aliases = pd.read_csv(aliases_file, sep="\t")
    # print(aliases.head())  # Columns: protein_id, alias
    # print(f"Number of rows in aliase: {aliases.shape[0]}") # 2035824

    merfish_genes = adata.var_names.tolist()
    # print(f"MERFISH genes: {merfish_genes[:5]}") 

    ### Use the aliases file to map MERFISH gene names to STRING protein IDs.###
    # Map MERFISH genes to STRING protein IDs
    mapped_proteins = aliases[aliases["alias"].isin(merfish_genes)]
    # print(mapped_proteins.head())  # Mapped protein IDs
    # print(f"Number of rows in aliase mapped: {mapped_proteins.shape[0]}") # 971

    ### Filter STRING Interactions for Mapped Proteins, Extract the subset of interactions relevant to the mapped proteins.###
    # Get the protein IDs from the mapping
    mapped_protein_ids = mapped_proteins["#string_protein_id"].tolist()
    # Filter interactions for relevant proteins
    filtered_interactions = interactions[
        (interactions["protein1"].isin(mapped_protein_ids)) & 
        (interactions["protein2"].isin(mapped_protein_ids))
    ]
    # print(filtered_interactions.head())  # Relevant protein interactions
    # print(f"Number of rows in filtered_interactions: {filtered_interactions.shape[0]}") # 32276

    ### Integrate STRING Data with MERFISH, Add the filtered STRING interaction data to the MERFISH dataset for downstream analysis ###
    # Add STRING interactions as an additional layer or annotation
    adata.uns["string_interactions"] = filtered_interactions

    # Add mapped protein IDs to the genes
    adata.var["string_protein_id"] = adata.var_names.map(
        lambda gene: mapped_proteins.loc[mapped_proteins["alias"] == gene, "#string_protein_id"].values[0]
        if gene in mapped_proteins["alias"].values else None
    )

    # Verify integration
    # print(adata.var.head())  # Check added column

    # # Create a graph from filtered interactions
    # G = nx.Graph()
    # for _, row in filtered_interactions.iterrows():
    #     G.add_edge(row["protein1"], row["protein2"], weight=row["combined_score"])

    # # Draw the network
    # plt.figure(figsize=(10, 10))
    # nx.draw(G, with_labels=False, node_size=10)
    # plt.savefig("graph.png", dpi=300)




if not Path(DATADIR / "STRING").exists():
    download_string_data()

# Paths to STRING files
interaction_file = DATADIR / "STRING/10090.protein.links.full.v12.0.txt"
info_file = DATADIR / "STRING/10090.protein.info.v12.0.txt"
aliases_file = DATADIR / "STRING/10090.protein.aliases.v12.0.txt"

adata = sc.read_h5ad(DATADIR / "expression_matrices/MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-log2.h5ad")
load_protein_protein_interaction_data(adata, interaction_file, info_file, aliases_file)
adata.write(DATADIR /"Intermediate/merfish_w_ppi.h5ad")
# print(adata.var.head()) 
#                    gene_symbol transcript_identifier         string_protein_id
# gene_identifier                                                               
# ENSMUSG00000026778       Prkcq    ENSMUST00000028118  10090.ENSMUSP00000028118
# ENSMUSG00000026837      Col5a1    ENSMUST00000028280  10090.ENSMUSP00000028280
# ENSMUSG00000001985       Grik3    ENSMUST00000030676  10090.ENSMUSP00000030676
# ENSMUSG00000039323      Igfbp2    ENSMUST00000047328  10090.ENSMUSP00000046610
# ENSMUSG00000048387        Osr1    ENSMUST00000057021  10090.ENSMUSP00000055486


# TODO: find protein-ligand interaction db, integration ppi with protein-ligand interaction data, construct a graph
### Protein-Ligand Interaction Data from bindingDB ###

