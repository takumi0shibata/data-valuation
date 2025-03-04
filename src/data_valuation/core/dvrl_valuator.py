import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import tempfile
from tqdm import tqdm
import wandb
from typing import Dict, List, Union, Tuple, Optional, Any, Literal

from ..core.base_valuator import BaseValuator
from ..models.mlp import MLP
from ..models.features_model import FeaturesModel
from ..models.dvrl import DataValueEstimator, DvrlLoss
from ..utils.training import fit_model, predict
from ..utils.metrics import calculate_metric

class DVRLValuator(BaseValuator):
    """DVRL (Data Valuation using Reinforcement Learning) data value estimator."""
    
    def __init__(
        self,
        prompt_id: int,
        device: str = "cpu",
        seed: int = 42,
        metric: Literal["mse", "qwk", "corr"] = "qwk",
        wandb_logging: bool = False,
        wandb_project: str = "data-valuation",
        wandb_name: Optional[str] = None,
        embedding_model: str = "microsoft/deberta-v3-large",
        max_length: int = 512,
        pooling_strategy: Literal["mean", "cls"] = "cls",
        hidden_dim: int = 100,
        comb_dim: int = 10,
        iterations: int = 1000,
        inner_iterations: int = 100,
        batch_size: int = 10000,
        learning_rate: float = 1e-3,
        batch_size_predictor: int = 512,
        model_type: str = "mlp",
        temp_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the DVRL valuator.
        
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
            pooling_strategy: Pooling strategy for the embedding model.
            hidden_dim: Hidden dimension for value estimator.
            comb_dim: Combination dimension for value estimator.
            iterations: Number of outer iterations.
            inner_iterations: Number of inner iterations for training.
            batch_size: Batch size for DVRL.
            learning_rate: Learning rate for optimizers.
            batch_size_predictor: Batch size for predictor training.
            model_type: Type of model to use ('mlp' or 'features').
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
        self.hidden_dim = hidden_dim
        self.comb_dim = comb_dim
        self.iterations = iterations
        self.inner_iterations = inner_iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.batch_size_predictor = batch_size_predictor
        self.model_type = model_type
        self.epsilon = 1e-8
        self.threshold = 0.9
        self.temp_dir = temp_dir or tempfile.mkdtemp()

        # Print hyperparameters
        print("\n=== DVRLValuator Hyperparameters ===")
        print(f"Prompt ID: {prompt_id}")
        print(f"Device: {device}")
        print(f"Seed: {seed}")
        print(f"Metric: {metric}")
        print(f"Embedding Model: {embedding_model}")
        print(f"Max Length: {max_length}")
        print(f"Pooling Strategy: {pooling_strategy}")
        print(f"Hidden Dimension: {hidden_dim}")
        print(f"Combination Dimension: {comb_dim}")
        print(f"Iterations: {iterations}")
        print(f"Inner Iterations: {inner_iterations}")
        print(f"Batch Size: {batch_size}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Predictor Batch Size: {batch_size_predictor}")
        print(f"Model Type: {model_type}")
        print(f"Epsilon: {self.epsilon}")
        print(f"Threshold: {self.threshold}")
        print("================================\n")

        # Initialize wandb if enabled
        if self.wandb_logging:
            config = {
                "valuator": "DVRLValuator",
                "prompt_id": prompt_id,
                "device": device,
                "seed": seed,
                "metric": metric,
                "embedding_model": embedding_model,
                "max_length": max_length,
                "pooling_strategy": pooling_strategy,
                "hidden_dim": hidden_dim,
                "comb_dim": comb_dim,
                "iterations": iterations,
                "inner_iterations": inner_iterations,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "batch_size_predictor": batch_size_predictor,
                "model_type": model_type,
                "epsilon": self.epsilon,
                "threshold": self.threshold,
                **kwargs
            }
            wandb.init(
                project=self.wandb_project,
                config=config,
                name=self.wandb_name or f"dvrl_valuator_prompt_{prompt_id}"
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
        Implementation of the value estimation logic using DVRL.
        
        Args:
            x_train: Training features.
            y_train: Training labels.
            x_val: Validation features.
            y_val: Validation labels.
            sample_ids: Unique identifiers for training samples.
                        
        Returns:
            A dictionary mapping sample IDs to estimated DVRL values.
        """
        if sample_ids is None:
            sample_ids = list(range(len(x_train)))
            
        if len(sample_ids) != len(x_train):
            raise ValueError("Number of sample IDs must match number of training samples")
        
        # Ensure y_train is 2D
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
            
        dvrl_values = self._train_dvrl(x_train, y_train, x_val, y_val)
        
        return {sample_ids[i]: dvrl_values[i] for i in range(len(sample_ids))}
    
    def _initialize_model(self, input_dim=None) -> nn.Module:
        """Initialize a new model instance."""
        if self.model_type == 'mlp':
            model = MLP(input_feature=input_dim)
        else:  # features model
            model = FeaturesModel()
            
        return model.to(self.device)
    
    def _train_dvrl(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[Any, float]:
        """
        Train the DVRL model and compute data values.
        
        Args:
            x_train: Training features.
            y_train: Training labels.
            x_val: Validation features.
            y_val: Validation labels.
            
        Returns:
            A dictionary mapping sample IDs to estimated DVRL values.
        """
        device = torch.device(self.device)
        data_dim = x_train.shape[1]
        label_dim = y_train.shape[1]
        
        # Create and save initial model
        init_model = self._initialize_model(data_dim)
        model_path = os.path.join(self.temp_dir, "init_model_dvrl.pth")
        torch.save(init_model.state_dict(), model_path)
        
        # Train original model on all data
        ori_model = self._initialize_model(data_dim)
        ori_model.load_state_dict(torch.load(model_path, weights_only=True))
        
        fit_model(
            ori_model,
            x_train,
            y_train,
            self.batch_size_predictor,
            self.inner_iterations,
            device,
            self.learning_rate
        )
        
        # Train validation model on validation data
        val_model = self._initialize_model(data_dim)
        val_model.load_state_dict(torch.load(model_path, weights_only=True))
        
        fit_model(
            val_model,
            x_val,
            y_val,
            self.batch_size_predictor,
            self.inner_iterations,
            device,
            self.learning_rate
        )
        
        # Initialize data value estimator
        value_estimator = DataValueEstimator(
            input_dim=data_dim + label_dim,  # x and y
            hidden_dim=self.hidden_dim,
            comb_dim=self.comb_dim,
            y_pred_diff_dim=label_dim,  # dimension of y_pred_diff
            layer_number=5,
            activation_fn=nn.Tanh()
        ).to(device)
        
        dvrl_criterion = DvrlLoss(self.epsilon, self.threshold).to(device)
        dvrl_optimizer = optim.Adam(value_estimator.parameters(), lr=self.learning_rate)
        
        # Compute validation prediction differences
        y_source_valid_pred = predict(
            val_model,
            x_train,
            self.batch_size_predictor,
            device
        )
        y_pred_diff = np.abs(y_train - y_source_valid_pred)
        
        # Compute baseline performance
        y_valid_hat = predict(
            ori_model,
            x_val,
            self.batch_size_predictor,
            device
        )
        baseline_perf = calculate_metric(y_val, y_valid_hat, self.prompt_id, self.metric)
        
        # Training loop
        for iteration in tqdm(range(self.iterations), desc="Training DVRL"):
            value_estimator.train()
            dvrl_optimizer.zero_grad()
            
            # Sample a batch
            batch_size = min(self.batch_size, x_train.shape[0])
            batch_indices = np.random.permutation(x_train.shape[0])[:batch_size]
            x_batch = torch.tensor(x_train[batch_indices], dtype=torch.float32).to(device)
            y_batch = torch.tensor(y_train[batch_indices], dtype=torch.float32).to(device)
            y_diff_batch = torch.tensor(y_pred_diff[batch_indices], dtype=torch.float32).to(device)
            
            # Generate selection probabilities
            est_dv_curr = value_estimator(x_batch, y_batch, y_diff_batch).squeeze()
            
            # Sample selection probabilities
            sel_prob_curr = np.random.binomial(1, est_dv_curr.detach().cpu().numpy())
            if np.sum(sel_prob_curr) == 0:
                # Avoid zero selection
                est_dv_curr = torch.full_like(est_dv_curr, 0.5)
                sel_prob_curr = np.random.binomial(1, est_dv_curr.detach().cpu().numpy())
            
            # Train a new model with selected data
            new_model = self._initialize_model(data_dim)
            new_model.load_state_dict(torch.load(model_path, weights_only=True))
            
            fit_model(
                new_model,
                x_batch,
                y_batch,
                self.batch_size_predictor,
                self.inner_iterations,
                device,
                self.learning_rate,
                sel_prob_curr
            )
            
            # Evaluate on validation set
            y_valid_hat = predict(
                new_model,
                x_val,
                self.batch_size_predictor,
                device
            )
            
            # Calculate performance
            dvrl_perf = calculate_metric(y_val, y_valid_hat, self.prompt_id, self.metric)
            
            # Calculate reward
            if self.metric == 'mse':
                reward = baseline_perf - dvrl_perf
            elif self.metric == 'qwk':
                reward = dvrl_perf - baseline_perf
            elif self.metric == 'corr':
                reward = dvrl_perf - baseline_perf
            
            # Update data value estimator
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)
            sel_prob_curr_tensor = torch.tensor(sel_prob_curr, dtype=torch.float32).to(device)
            
            loss = dvrl_criterion(est_dv_curr, sel_prob_curr_tensor, reward_tensor)
            loss.backward()
            dvrl_optimizer.step()
            
            if self.wandb_logging:
                wandb.log({
                    "iteration": iteration,
                    "reward": reward,
                    "dvrl_loss": loss.item(),
                    "max_prob": est_dv_curr.max().item(),
                    "min_prob": est_dv_curr.min().item(),
                    f"{self.metric}": dvrl_perf
                })
                
        # Get final data values for all training points
        value_estimator.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
            y_diff_tensor = torch.tensor(y_pred_diff, dtype=torch.float32).to(device)
            
            data_values = value_estimator(x_tensor, y_tensor, y_diff_tensor).squeeze()

        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)

        # Finalize wandb run
        if self.wandb_logging:
            wandb.finish()
        
        # Convert numpy array to dictionary with float values
        data_values_dict = {i: float(v) for i, v in enumerate(data_values.cpu().numpy())}
        return data_values_dict