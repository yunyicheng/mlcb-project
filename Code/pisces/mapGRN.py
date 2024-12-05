import pandas as pd
import scanpy as sc
import requests
from pathlib import Path
import requests

from .config import *

# download GRN data
def download_trrust_data(url, output_file):
    try:
        # Send HTTP GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        # Ensure the output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        # Save the content to a file
        with open(output_file, "wb") as file:
            file.write(response.content)
        print(f"TRRUST data downloaded successfully: {output_file}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

def load_data():
    # URLs for TRRUST data
    mouse_url = "http://www.grnpedia.org/trrust/data/trrust_rawdata.mouse.tsv"
    if not Path(DATADIR / "GRN/trrust_mouse.tsv").exists():
        download_trrust_data(mouse_url, DATADIR / "GRN/trrust_mouse.tsv")

def map_grn():
    # Load MERFISH data and TRRUST data
    adata = sc.read_h5ad(DATADIR / "expression_matrices/MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-log2.h5ad")
    grn_data = pd.read_csv(DATADIR / "GRN/trrust_mouse.tsv", sep="\t", header=None, 
                        names=["TF", "Target", "Interaction", "PMID"])
    # TF: Transcription factor.
    # Target: Target gene regulated by the TF.
    # Interaction: Type of interaction (activation or repression).
    # PMID: PubMed ID of the supporting publication.

    # print(adata.obs.head())  # Cell metadata
    # print(adata.var.head())  # Gene metadata
    # print(grn_data.head())  # GRN data

    # print(f"Number of genes in merfish: {adata.shape[1]}") # 550
    # print(f"Number of rows in grn_data: {grn_data.shape[0]}") # 7057


    # Extract the mapping of gene_identifier to gene_symbol
    gene_mapping = adata.var["gene_symbol"].to_dict()
    # Replace TF and Target in GRN data with gene_identifier
    grn_data["TF_id"] = grn_data["TF"].map({v: k for k, v in gene_mapping.items()})
    grn_data["Target_id"] = grn_data["Target"].map({v: k for k, v in gene_mapping.items()})

    # Drop rows where mapping failed (non-overlapping genes)
    grn_data = grn_data.dropna(subset=["TF_id", "Target_id"])

    # print(f"Number of rows in grn_data after map: {grn_data.shape[0]}") # 83

    # # Inspect the mapped GRN data
    # print(grn_data.head())

    # Save grn_data to a TSV file"
    grn_data.to_csv(INTERDATADIR / "grn_mapped.tsv", sep="\t", index=False)
    return grn_data

    # Map regulatory network data to adata.var based on 'Target_id'
    # adata.var["Regulatory_TF"] = adata.var.index.map(grn_data.set_index("Target_id")["TF"])
    # adata.var["Regulatory_TF_id"] = adata.var.index.map(grn_data.set_index("Target_id")["TF_id"])
    # adata.var["Interaction_Type"] = adata.var.index.map(grn_data.set_index("Target_id")["Interaction"])
    # adata.var["PMID"] = adata.var.index.map(grn_data.set_index("Target_id")["PMID"])


    # print(adata.var[["Regulatory_TF", "Interaction_Type", "PMID"]].head())

## SUMMARY: download GRN data from TRRUST, map gene identifiers to gene symbols, 
# and save the mapped data to a TSV file.