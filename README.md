# Cell-Cell Communication Analysis Using Graph Attention Networks (GAT)

## Overview
This project uses a **Graph Attention Network (GAT)** to study **cell-cell communication** by predicting coexpression scores for ligand-receptor interactions. By integrating spatial and transcriptomic data, the model captures how spatial relationships influence transcriptional interactions and cell communication.

## Purpose
The goal of this project is to:
- Predict coexpression scores for ligand-receptor interactions (ligand-receptor edges).
- Leverage spatial relationships (spatial edges) to provide additional context for the analysis.
- Develop a framework to analyze cell-cell communication within spatial environments using single-cell spatial transcriptomics data.

---

## Model Structure

### Graph Attention Network (GAT)
The model consists of:
1. **Node Embedding Layers**:
   - Two stacked GAT layers generate node embeddings using the graph structure and node features.
   - These embeddings are shared across all edge computations.

2. **Edge-Specific Layers**:
   - **Ligand-receptor edges**:
     - Coexpression scores are predicted using a fully connected layer after processing edge embeddings.
   - **Spatial edges**:
     - Distances are incorporated into edge embeddings for additional context but are not used as prediction targets.

3. **Output Layer**:
   - A fully connected layer outputs predicted coexpression scores for ligand-receptor edges.

This modular design allows the model to focus on biologically relevant relationships while leveraging spatial context for improved accuracy.

---

## Training Data

### Input Graphs
Graphs are constructed from **single-cell spatial transcriptomics datasets**. Each graph contains:

1. **Nodes**:
   - Represent cells characterized by their transcriptomic profiles (gene expression levels).
   - Node features are derived from highly variable genes selected during preprocessing.

2. **Edges**:
   - **Ligand-receptor edges**:
     - Represent transcriptional coexpression scores for specific ligand-receptor pairs.
     - Serve as both input features and prediction targets.
   - **Spatial edges**:
     - Represent physical distances between cells based on spatial coordinates.
     - Serve as input features to contextualize ligand-receptor interactions but are not prediction targets.

### Graph Format
The input graphs are stored using **PyTorch Geometric** objects, which include:
- `x`: Node features (gene expression profiles).
- `edge_index` and `edge_attr`: Ligand-receptor edges (coexpression scores).
- `spatial_edge_index` and `spatial_edge_attr`: Spatial edges (distances).

---

## Training Techniques

### Objective
- **Self-Supervised Learning**: The model is trained to predict coexpression scores for masked ligand-receptor edges.

### Techniques
1. **Masking Strategy**:
   - A subset of ligand-receptor edges is randomly masked during training.
   - The model learns to reconstruct the coexpression scores for these edges.

2. **Loss Function**:
   - **Mean Squared Error (MSE)** is used to compute the difference between predicted and true coexpression scores.

3. **Optimization Techniques**:
   - **Cyclical Learning Rate Scheduler (CLR)**:
     - Dynamically adjusts the learning rate between a minimum and maximum value to improve convergence and prevent overfitting.
   - **Stochastic Weight Averaging (SWA)**:
     - Maintains an averaged copy of the model weights during training, improving generalization.

4. **Batch Processing**:
   - Graphs are batched together with separate handling for ligand-receptor and spatial edges to ensure compatibility with the model.

---

## Model Training

### Training Process
The training function processes batches of graphs, performing the following steps:
1. Load graphs with nodes, ligand-receptor edges, and spatial edges.
2. Mask a subset of ligand-receptor edges and train the model to predict their coexpression scores.
3. Incorporate spatial edges into the model as contextual features during training.
