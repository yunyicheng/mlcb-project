import pandas as pd
import scanpy as sc
import requests
import os
from pathlib import Path
import requests
import zipfile
import shutil
import networkx as nx
import matplotlib.pyplot as plt

from config import *


import requests
import os

# TODO: find protein-ligand interaction db, integration ppi with protein-ligand interaction data, construct a graph
### Protein-Ligand Interaction Data from bindingDB ###

def download_data():
    url = "https://www.bindingdb.org/bind/BDB-Oracle_All_202412_dmp.zip"
    output_file = Path(DATADIR)/"bindingDB/BDB-Oracle_All_202412_dmp.zip"
    os.makedirs(output_file.parent, exist_ok=True)

    print(f"Downloading BindingDB full dataset from {url}...")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {output_file}")
    else:
        print(f"Failed to download BindingDB dataset. Status code: {response.status_code}")

def unzip_data():
    # Define the path to the zip file and the extraction directory

    zip_file = Path(DATADIR)/"bindingDB/BDB-Oracle_All_202412_dmp.zip"
    extraction_dir = Path(DATADIR)/"bindingDB/"
    # Ensure the extraction directory exists
    # Extract the zip file
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extraction_dir)

    print(f"Extracted {zip_file} to {extraction_dir}")


if not Path(DATADIR / "bindingDB").exists():
    download_data()
    unzip_data()


# Extract PPI data from AnnData

adata = sc.read_h5ad((DATADIR /"Intermediate/merfish_w_ppi.h5ad"))

ppi_data = pd.DataFrame(adata.uns["string_interactions"])  # PPI interactions
string_protein_map = adata.var["string_protein_id"]  # STRING protein mapping
print(ppi_data.head())

import subprocess

def import_oracle_dump(parfile_path):
    """
    Imports an Oracle dump file using impdp and a parameter file.
    
    Args:
        parfile_path (str): Path to the `imp_all.txt` parameter file.
        
    Returns:
        None
    """
    try:
        # Command to run impdp
        command = ["impdp", f"parfile={parfile_path}"]
        
        # Execute the command
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Output the results
        if process.returncode == 0:
            print("Import completed successfully.")
            print("Output:\n", process.stdout)
        else:
            print("Import failed.")
            print("Error:\n", process.stderr)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Path to the parameter file
parameter_file = "imp_all.txt"

# Run the import function
import_oracle_dump(parameter_file)

import cx_Oracle

def verify_import():
    connection = cx_Oracle.connect("BIND_2000_05", "xxxxxxxxxxxx", "localhost/XEPDB1")
    cursor = connection.cursor()
    
    # Example query to check imported data
    cursor.execute("SELECT COUNT(*) FROM imported_table_name")
    row_count = cursor.fetchone()[0]
    print(f"Number of rows in the imported table: {row_count}")
    
    cursor.close()
    connection.close()

verify_import()