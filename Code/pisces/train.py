from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
from torch.optim.swa_utils import SWALR, AveragedModel
from torch_geometric.loader import DataLoader

from .config import *
from .model import *


def train_gat_with_spatial_edges(
    model, data_list, num_epochs=100, learning_rate=0.01, mask_ratio=0.1, batch_size=4
):
    # Create DataLoader for batching graphs
    data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_true = []
        all_predicted = []

        for batch in data_loader:
            optimizer.zero_grad()

            # Get batch components
            x = batch.x
            ligand_edge_index = batch.edge_index
            ligand_edge_attr = batch.edge_attr[:, 0].unsqueeze(1)  # Coexpression scores
            spatial_edge_index = batch.spatial_edge_index
            spatial_edge_attr = batch.spatial_edge_attr  # Distances

            # Randomly mask a subset of ligand-receptor edges
            num_edges = ligand_edge_index.size(1)
            num_masked = int(mask_ratio * num_edges)
            perm = torch.randperm(num_edges)
            mask = perm[:num_masked]
            masked_edge_index = ligand_edge_index[:, mask]
            masked_coexpression = ligand_edge_attr[mask]

            # Forward pass with both edge types
            edge_predictions = model(
                x,
                ligand_edge_index,
                ligand_edge_attr,
                spatial_edge_index=spatial_edge_index,
                spatial_edge_attr=spatial_edge_attr,
            )

            # Compute loss (reconstruction of coexpression scores only)
            loss = criterion(edge_predictions[mask], masked_coexpression)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Collect true and predicted values for metrics
            all_true.append(masked_coexpression.detach().cpu())
            all_predicted.append(edge_predictions[mask].detach().cpu())

        # Concatenate all true and predicted values
        all_true = torch.cat(all_true).numpy()
        all_predicted = torch.cat(all_predicted).numpy()

        # Calculate additional metrics
        mae = mean_absolute_error(all_true, all_predicted)
        pearson_corr, _ = pearsonr(all_true.flatten(), all_predicted.flatten())
        spearman_corr, _ = spearmanr(all_true.flatten(), all_predicted.flatten())

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, "
            f"MAE: {mae:.4f}, Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}"
        )

    return model


def train_model():
    # Load all graphs
    train_data = load_graphs_from_pickle(GRAPHOUTDIR)

    # Check the structure of the first graph
    print(f"Node features: {train_data[0].x.size()}")
    print(f"Ligand-receptor edge features: {train_data[0].edge_attr.size()}")
    print(f"Spatial edge features: {train_data[0].spatial_edge_attr.size()}")

    # Initialize the GAT model
    model = GATWithSpatialEdges(
        in_channels=train_data[0].x.size(1),
        hidden_channels=64,
        out_channels=1,
    )

    # Train the model on all graphs
    trained_model = train_gat_with_spatial_edges(
        model=model,
        data_list=train_data,
        num_epochs=100,
        learning_rate=0.01,
        mask_ratio=0.1,
        batch_size=4,
    )

    return trained_model
