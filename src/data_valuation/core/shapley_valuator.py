import numpy as np
import torch
import torch.nn as nn
from collections import deque
import wandb
from typing import Dict, List, Union, Tuple, Optional, Any, Literal

from ..core.base_valuator import BaseValuator
from ..models.mlp import MLP
from ..models.features_model import FeaturesModel

class ShapleyValuator(BaseValuator):
    """Data Shapley value estimator."""
    
    def __init__(
        self,
        prompt_id: int,
        device: str = "cpu",
        seed: int = 42,
        metric: Literal["mse", "qwk", "corr"] = "mse",
        wandb_logging: bool = False,
        wandb_project: str = "data-valuation",
        wandb_name: Optional[str] = None,
        embedding_model: str = "microsoft/deberta-v3-large",
        max_length: int = 512,
        max_iter: int = 5000,
        threshold: float = 0.05,
        model_type: str = "mlp",
        learning_rate: float = 1e-3,
        **kwargs
    ):
        """
        Initialize the Data Shapley valuator.
        
        Args:
            prompt_id: Prompt ID for the task.
            device: Device to run computations on.
            seed: Random seed for reproducibility.
            metric: Evaluation metric.
            wandb_logging: Whether to log to Weights & Biases.
            wandb_project: Name of the W&B project.
            wandb_name: Name of the W&B run. If None, a default name will be used.
            embedding_model: Name or path of the pre-trained model for text embedding.
            max_length: Maximum sequence length for tokenization.
            max_iter: Maximum number of iterations.
            threshold: Convergence threshold.
            model_type: Type of model to use ('mlp' or 'features').
            learning_rate: Learning rate for optimizer.
            **kwargs: Additional arguments.
        """
        super().__init__(
            prompt_id=prompt_id,
            device=device,
            seed=seed,
            metric=metric,
            wandb_logging=wandb_logging,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
            embedding_model=embedding_model,
            max_length=max_length,
            **kwargs
        )
        self.max_iter = max_iter
        self.threshold = threshold
        self.model_type = model_type
        self.learning_rate = learning_rate

        # Print hyperparameters
        print("\n=== ShapleyValuator Hyperparameters ===")
        print(f"Prompt ID: {prompt_id}")
        print(f"Device: {device}")
        print(f"Seed: {seed}")
        print(f"Metric: {metric}")
        print(f"Embedding Model: {embedding_model}")
        print(f"Max Length: {max_length}")
        print(f"Max Iterations: {max_iter}")
        print(f"Threshold: {threshold}")
        print(f"Model Type: {model_type}")
        print(f"Learning Rate: {learning_rate}")
        print("================================\n")

        # Initialize wandb if enabled
        if self.wandb_logging:
            config = {
                "valuator": "ShapleyValuator",
                "prompt_id": prompt_id,
                "device": device,
                "seed": seed,
                "metric": metric,
                "embedding_model": embedding_model,
                "max_length": max_length,
                "max_iter": max_iter,
                "threshold": threshold,
                "model_type": model_type,
                "learning_rate": learning_rate,
                **kwargs
            }
            wandb.init(
                project=self.wandb_project,
                config=config,
                name=self.wandb_name or f"shapley_valuator_prompt_{prompt_id}"
            )
    
    def _estimate_values_impl(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        sample_ids: Optional[List[Any]] = None
    ) -> Dict[Any, float]:
        """
        Implementation of the value estimation logic using Data Shapley.
        
        Args:
            x_train: Training features.
            y_train: Training labels.
            x_val: Validation features.
            y_val: Validation labels.
            sample_ids: Unique identifiers for training samples.
                        
        Returns:
            A dictionary mapping sample IDs to estimated Shapley values.
        """
        if sample_ids is None:
            sample_ids = list(range(len(x_train)))
            
        if len(sample_ids) != len(x_train):
            raise ValueError("Number of sample IDs must match number of training samples")
            
        shapley_values = self._compute_data_shapley(x_train, y_train, x_val, y_val)
        
        return {sample_ids[i]: shapley_values[i] for i in range(len(sample_ids))}
    
    def _compute_data_shapley(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray
    ) -> np.ndarray:
        """
        Compute Data Shapley values.
        
        Args:
            x_train: Training features.
            y_train: Training labels.
            x_val: Validation features.
            y_val: Validation labels.
            
        Returns:
            An array of Shapley values.
        """
        n_samples = x_train.shape[0]
        input_dim = x_train.shape[1]
        shapley_values = np.zeros(n_samples)
        past_shapley_values = deque(maxlen=100)
        t = 0
        
        device = torch.device(self.device)
        criterion = nn.MSELoss()
        
        # Ensure validation data has correct shape
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
        if len(y_val_tensor.shape) == 1:
            y_val_tensor = y_val_tensor.unsqueeze(1)
        
        while t < self.max_iter:
            t += 1
            permutation = np.random.permutation(n_samples)
            
            # Initialize model
            if self.model_type == 'mlp':
                model = MLP(input_feature=input_dim).to(device)
            else:  # features model
                model = FeaturesModel().to(device)
                
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            losses = []
            
            # Initial validation loss
            model.eval()
            with torch.no_grad():
                y_pred = model(x_val_tensor)
                if y_pred.shape != y_val_tensor.shape:
                    y_pred = y_pred.view(y_val_tensor.shape)
                init_loss = criterion(y_pred, y_val_tensor).item()
            losses.append(init_loss)
            
            # Iterate through training points in the permutation
            for j in range(n_samples):
                x_train_point = torch.tensor(x_train[permutation[j]].reshape(1, -1), dtype=torch.float32).to(device)
                y_train_point = torch.tensor(y_train[permutation[j]], dtype=torch.float32).to(device)
                if len(y_train_point.shape) == 1:
                    y_train_point = y_train_point.unsqueeze(1)
                
                # Train on single point
                model.train()
                optimizer.zero_grad()
                y_pred = model(x_train_point)
                if y_pred.shape != y_train_point.shape:
                    y_pred = y_pred.view(y_train_point.shape)
                loss = criterion(y_pred, y_train_point)
                loss.backward()
                optimizer.step()
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    y_pred = model(x_val_tensor)
                    if y_pred.shape != y_val_tensor.shape:
                        y_pred = y_pred.view(y_val_tensor.shape)
                    val_loss = criterion(y_pred, y_val_tensor).item()
                
                # Update Shapley value
                shapley_values[permutation[j]] = (t - 1) / t * shapley_values[permutation[j]] + 1 / t * (val_loss - losses[-1])
                losses.append(val_loss)
            
            # Check for convergence
            if t > 100:
                past_shapley_value = past_shapley_values.popleft()
                convergence_criteria = np.mean(np.abs(shapley_values - past_shapley_value) / (np.abs(shapley_values) + 1e-10))
                print(f"Iteration {t}: {convergence_criteria}")
                
                if convergence_criteria < self.threshold:
                    print(f"Early stopping at iteration {t}")
                    break
            else:
                print(f"Iteration {t}")
                
            past_shapley_values.append(shapley_values.copy())
            
            # Log to wandb if enabled
            if self.wandb_logging:
                wandb.log({
                    "iteration": t,
                    "convergence_criteria": convergence_criteria if t > 100 else None,
                    "shapley_mean": np.mean(shapley_values),
                    "shapley_std": np.std(shapley_values)
                })
                
        return shapley_values