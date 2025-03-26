import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import os
import tempfile
import wandb
from typing import Dict, List, Union, Tuple, Optional, Any, Literal

from ..core.base_valuator import BaseValuator
from ..models.mlp import MLP
from ..models.features_model import FeaturesModel
from ..utils.training import fit_model, predict

class LOOValuator(BaseValuator):
    """Leave-One-Out data value estimator."""
    
    def __init__(
        self,
        prompt_id: int,
        device: str = "cpu",
        seed: int = 42,
        metric: Literal["mse", "qwk", "corr"] = "mse", # TODO: suport other metrics (qwk, corr)
        wandb_logging: bool = False,
        wandb_project: str = "data-valuation",
        wandb_name: Optional[str] = None,
        embedding_model: str = "microsoft/deberta-v3-large",
        max_length: int = 512,
        pooling_strategy: Literal["mean", "cls"] = "cls",
        batch_size: int = 512,
        epochs: int = 100,
        model_type: str = "mlp",
        learning_rate: float = 1e-3,
        temp_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Leave-One-Out valuator.
        
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
            pooling_strategy: Pooling strategy for text embedding.
            batch_size: Batch size for training.
            epochs: Number of training epochs.
            model_type: Type of model to use ('mlp' or 'features').
            learning_rate: Learning rate for optimizer.
            temp_dir: Directory for saving temporary files.
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
            pooling_strategy=pooling_strategy,
            **kwargs
        )
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.temp_dir = temp_dir or tempfile.mkdtemp()

        # Print hyperparameters
        print("\n=== LOOValuator Hyperparameters ===")
        print(f"Prompt ID: {prompt_id}")
        print(f"Device: {device}")
        print(f"Seed: {seed}")
        print(f"Metric: {metric}")
        print(f"Embedding Model: {embedding_model}")
        print(f"Max Length: {max_length}")
        print(f"Pooling Strategy: {pooling_strategy}")
        print(f"Batch Size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Model Type: {model_type}")
        print(f"Learning Rate: {learning_rate}")
        print("================================\n")

        # Initialize wandb if enabled
        if self.wandb_logging:
            config = {
                "valuator": "LOOValuator",
                "prompt_id": prompt_id,
                "device": device,
                "seed": seed,
                "metric": metric,
                "embedding_model": embedding_model,
                "max_length": max_length,
                "pooling_strategy": pooling_strategy,
                "batch_size": batch_size,
                "epochs": epochs,
                "model_type": model_type,
                "learning_rate": learning_rate,
                **kwargs
            }
            wandb.init(
                project=self.wandb_project,
                config=config,
                name=self.wandb_name or f"loo_valuator_prompt_{prompt_id}"
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
        Implementation of the value estimation logic using Leave-One-Out.
        
        Args:
            x_train: Training features.
            y_train: Training labels.
            x_val: Validation features.
            y_val: Validation labels.
            sample_ids: Unique identifiers for training samples.
                        
        Returns:
            A dictionary mapping sample IDs to estimated LOO values.
        """
        if sample_ids is None:
            sample_ids = list(range(len(x_train)))
            
        if len(sample_ids) != len(x_train):
            raise ValueError("Number of sample IDs must match number of training samples")
            
        loo_values = self._compute_loo_values(x_train, y_train, x_val, y_val)
        
        return {sample_ids[i]: loo_values[i] for i in range(len(sample_ids))}
    
    def _initialize_model(self, input_dim: int) -> nn.Module:
        """Initialize a new model instance."""
        if self.model_type == 'mlp':
            model = MLP(input_feature=input_dim)
        else:  # features model
            model = FeaturesModel()
            
        return model.to(self.device)
    
    def _compute_loo_values(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray
    ) -> np.ndarray:
        """
        Compute Leave-One-Out values.
        
        Args:
            x_train: Training features.
            y_train: Training labels.
            x_val: Validation features.
            y_val: Validation labels.
            
        Returns:
            An array of LOO values.
        """
        device = torch.device(self.device)
        n_samples = len(x_train)
        input_dim = x_train.shape[1] if len(x_train.shape) > 1 else 1
        
        # Create and save initial model
        init_model = self._initialize_model(input_dim)
        model_path = os.path.join(self.temp_dir, "init_model.pth")
        torch.save(init_model.state_dict(), model_path)
        
        # Train baseline model on all data
        baseline_model = self._initialize_model(input_dim)
        baseline_model.load_state_dict(torch.load(model_path, weights_only=True))
        
        fit_model(
            baseline_model,
            x_train,
            y_train,
            self.batch_size,
            self.epochs,
            device,
            self.learning_rate
        )
        
        # Evaluate baseline model
        baseline_preds = predict(
            baseline_model,
            x_val,
            self.batch_size,
            device
        )
        baseline_loss = mean_squared_error(y_val, baseline_preds)
        
        # Calculate LOO values
        loo_values = []
        for i in tqdm(range(n_samples), desc="Computing LOO values"):
            # Exclude i-th sample
            x_train_loo = np.delete(x_train, i, axis=0)
            y_train_loo = np.delete(y_train, i, axis=0)
            
            # Initialize model from scratch
            loo_model = self._initialize_model(input_dim)
            loo_model.load_state_dict(torch.load(model_path, weights_only=True))
            
            # Train on reduced dataset
            fit_model(
                loo_model,
                x_train_loo,
                y_train_loo,
                self.batch_size,
                self.epochs,
                device,
                self.learning_rate
            )
            
            # Evaluate
            loo_preds = predict(
                loo_model,
                x_val,
                self.batch_size,
                device
            )
            loo_loss = mean_squared_error(y_val, loo_preds)
            
            # Value = difference in performance (higher = more important)
            # Higher LOO loss means removing this sample hurts performance
            loo_value = loo_loss - baseline_loss
            loo_values.append(loo_value)
            
            # Log to wandb if enabled
            if self.wandb_logging and (i % 10 == 0 or i == n_samples - 1):
                wandb.log({
                    "sample_index": i,
                    "loo_value": loo_value,
                    "baseline_loss": baseline_loss,
                    "loo_loss": loo_loss
                })
                
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Finalize wandb run
        if self.wandb_logging:
            wandb.finish()
            
        return np.array(loo_values)