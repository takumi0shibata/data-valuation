import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import torch.nn as nn


def fit_model(
        model: nn.Module,
        x_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int,
        epochs: int,
        device: torch.device,
        learning_rate: float = 0.001,
        sample_weight: np.ndarray = None
) -> list:
    """
    Fit the model with the given data.
    Args:
        model: Model to train
        x_train: Training data
        y_train: Training labels
        batch_size: Batch size
        epochs: Number of epochs
        device: Device to run the model
        learning_rate: Learning rate for optimizer
        sample_weight: Sample weight for each data
    Returns:
        list: Loss history
    """
    
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Ensure x_train has correct shape
    if len(x_train.shape) == 1:
        x_train = x_train.reshape(-1, 1)
    
    # Convert to tensors and ensure consistent dimensions
    if not isinstance(x_train, torch.Tensor):
        x_train = torch.tensor(x_train, dtype=torch.float)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.float)
    
    # Ensure y_train has shape (batch_size, 1)
    if len(y_train.shape) == 1:
        y_train = y_train.unsqueeze(1)

    x_train = x_train.clone().detach().to(device)
    y_train = y_train.clone().detach().to(device)

    if sample_weight is not None:
        if not isinstance(sample_weight, torch.Tensor):
            sample_weight = torch.tensor(sample_weight, dtype=torch.float)
        if len(sample_weight.shape) == 1:
            sample_weight = sample_weight.unsqueeze(1)
        sample_weight = sample_weight.clone().detach().to(device)
        dataset = TensorDataset(x_train, y_train, sample_weight)
        loss_fn = nn.MSELoss(reduction='none')
    else:
        dataset = TensorDataset(x_train, y_train)
        loss_fn = nn.MSELoss(reduction='mean')

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if sample_weight is not None:
                x_batch, y_batch, w_batch = batch
                y_pred = model(x_batch)
                # Ensure predictions have same shape as targets
                if y_pred.shape != y_batch.shape:
                    y_pred = y_pred.view(y_batch.shape)
                loss = (loss_fn(y_pred, y_batch) * w_batch).mean()
            else:
                x_batch, y_batch = batch
                y_pred = model(x_batch)
                # Ensure predictions have same shape as targets
                if y_pred.shape != y_batch.shape:
                    y_pred = y_pred.view(y_batch.shape)
                loss = loss_fn(y_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        history.append(epoch_loss / len(train_loader))

    return history

def predict(
        model: nn.Module,
        x_test: np.ndarray,
        batch_size: int,
        device: torch.device
    ) -> np.ndarray:
    """
    Predict with the given model.
    Args:
        model: Model to predict
        x_test: Test data
        batch_size: Batch size
        device: Device to run the model
    Returns:
        np.ndarray: Predicted results with shape (n_samples, 1)
    """
    model = model.to(device)
    model.eval()

    # Ensure x_test has correct shape
    if len(x_test.shape) == 1:
        x_test = x_test.reshape(-1, 1)

    x_test = torch.tensor(x_test, dtype=torch.float)
    test_data = TensorDataset(x_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for x_batch in test_loader:
            x_batch = x_batch[0].to(device)
            y_pred = model(x_batch)
            # Ensure predictions have shape (batch_size, 1)
            if len(y_pred.shape) == 1:
                y_pred = y_pred.unsqueeze(1)
            preds.append(y_pred.cpu().numpy())
    return np.vstack(preds)