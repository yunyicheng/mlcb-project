from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
from torch.optim.swa_utils import SWALR, AveragedModel
from torch_geometric.loader import DataLoader

from .config import *
from .model import *


def train_gat_self_supervised(
    model, data_list, num_epochs=100, learning_rate=0.01, mask_ratio=0.1, batch_size=4
):
    # Create DataLoader for batching graphs
    data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    criterion = nn.MSELoss()

    # Set up Cyclical Learning Rate (CLR)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=learning_rate * 0.1,
        max_lr=learning_rate,
        step_size_up=10,
        mode="triangular",
    )

    # Set up Stochastic Weight Averaging (SWA)
    swa_model = AveragedModel(model)
    swa_start = int(num_epochs * 0.75)  # Start SWA at 75% of training
    swa_scheduler = SWALR(optimizer, swa_lr=learning_rate * 0.1)

    # Metrics storage
    metrics_log = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_true = []
        all_predicted = []

        for batch in data_loader:
            optimizer.zero_grad()

            # Get batch components
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr  # edge_attr contains coexpression + distance
            batch_mask = batch.batch

            # Separate coexpression scores and distances
            coexpression_scores = edge_attr[:, 0].unsqueeze(
                1
            )  # First column: coexpression
            distances = edge_attr[:, 1].unsqueeze(1)  # Second column: distance

            # Randomly mask a subset of edges
            num_edges = edge_index.size(1)
            num_masked = int(mask_ratio * num_edges)
            perm = torch.randperm(num_edges)
            mask = perm[:num_masked]
            masked_edge_index = edge_index[:, mask]
            masked_coexpression = coexpression_scores[mask]

            # Forward pass: Use both coexpression and distance as input features
            edge_attr_combined = torch.cat([coexpression_scores, distances], dim=1)
            edge_predictions = model(x, edge_index, edge_attr_combined, edge_mask=mask)

            # Compute loss (reconstruction of coexpression scores only)
            loss = criterion(edge_predictions, masked_coexpression)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Step the cyclical learning rate scheduler
            scheduler.step()

            total_loss += loss.item()

            # Collect true and predicted values for metrics
            all_true.append(masked_coexpression.detach().cpu())
            all_predicted.append(edge_predictions.detach().cpu())

        # Update SWA model and SWA scheduler
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # Concatenate all true and predicted values
        all_true = torch.cat(all_true).numpy()
        all_predicted = torch.cat(all_predicted).numpy()

        # Calculate additional metrics
        mae = mean_absolute_error(all_true, all_predicted)

        # Log metrics
        metrics_log.append(
            {
                "Epoch": epoch + 1,
                "MSE Loss": total_loss / len(data_loader),
                "MAE": mae,
            }
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {total_loss / len(data_loader):.4f}, "
            f"MAE: {mae:.4f}"
        )

    # Update BatchNorm statistics for SWA model
    torch.optim.swa_utils.update_bn(data_loader, swa_model)

    return swa_model, metrics_log


def train_model():
    # Load all graphs
    train_data = load_graphs_from_pickle(GRAPHOUTDIR)
    print(train_data[0].edge_attr.size(1))
    # Initialize the GAT model
    model = GATSelfSupervised(
        in_channels=train_data[0].x.size(1),
        hidden_channels=64,
        out_channels=train_data[0].edge_attr.size(1),
    )

    # Train the model on all graphs
    trained_model = train_gat_self_supervised(
        model=model,
        data_list=train_data,
        num_epochs=100,
        learning_rate=0.01,
        mask_ratio=0.5,
        batch_size=4,
    )

    return trained_model
