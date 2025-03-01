import numpy as np
from abc import ABC, abstractmethod
import torch
from typing import Dict, List, Union, Tuple, Optional, Any, Literal
from transformers import AutoTokenizer, AutoModel

class BaseValuator(ABC):
    """Abstract base class for all data valuators."""
    
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
        **kwargs
    ):
        """
        Initialize the base valuator.
        
        Args:
            device: Device to run computations on ('cpu' or 'cuda').
            seed: Random seed for reproducibility.
            metric: Evaluation metric ('mse', 'qwk', or 'corr').
            wandb_logging: Whether to log to Weights & Biases.
            wandb_project: Name of the W&B project.
            wandb_name: Name of the W&B run. If None, a default name will be used.
            embedding_model: Name or path of the pre-trained model for text embedding.
            max_length: Maximum sequence length for tokenization.
            **kwargs: Additional valuator-specific arguments.
        """
        self.prompt_id = prompt_id
        self.device = device
        self.seed = seed
        self.metric = metric
        self.wandb_logging = wandb_logging
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.config = kwargs
        self.embedding_model = embedding_model
        self.max_length = max_length
        
        # Initialize tokenizer and model for embeddings
        self.tokenizer = None
        self.model = None
        self._setup_seed()
        
    def _setup_seed(self):
        """Set random seeds for reproducibility."""
        import random
        import numpy as np
        import torch
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def _setup_embedding_model(self):
        """Initialize the embedding model and tokenizer if not already initialized."""
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model, use_fast=False)
            self.model = AutoModel.from_pretrained(self.embedding_model).to(self.device)
            self.model.eval()
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to embeddings using the specified model.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            numpy.ndarray: Array of embeddings with shape (n_samples, embedding_dim)
        """
        self._setup_embedding_model()
        
        embeddings = []
        batch_size = 32  # Adjust based on available memory
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                # Use mean pooling to get sentence embeddings
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                embeddings.append(sentence_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def estimate_values(
        self,
        x_train: Union[np.ndarray, List[str]],
        y_train: np.ndarray,
        x_val: Union[np.ndarray, List[str]],
        y_val: np.ndarray,
        sample_ids: Optional[List[Any]] = None
    ) -> Dict[Any, float]:
        """
        Estimate the value of each training sample.
        
        Args:
            x_train: Training features or raw texts.
            y_train: Training labels.
            x_val: Validation features or raw texts.
            y_val: Validation labels.
            sample_ids: Unique identifiers for training samples. If not provided,
                        indices will be used.
                        
        Returns:
            A dictionary mapping sample IDs to estimated values.
        """
        # Convert texts to embeddings if input is List[str]
        if isinstance(x_train, list) and isinstance(x_train[0], str):
            x_train = self.get_embeddings(x_train)
        if isinstance(x_val, list) and isinstance(x_val[0], str):
            x_val = self.get_embeddings(x_val)
            
        return self._estimate_values_impl(x_train, y_train, x_val, y_val, sample_ids)
    
    @abstractmethod
    def _estimate_values_impl(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        sample_ids: Optional[List[Any]] = None
    ) -> Dict[Any, float]:
        """
        Implementation of the value estimation logic.
        This method should be implemented by subclasses.
        """
        pass
    
    def save_values(self, values: Dict[Any, float], output_path: str):
        """
        Save the estimated values to a file.
        
        Args:
            values: Dictionary mapping sample IDs to estimated values.
            output_path: Path to save the values.
        """
        import json
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert any non-serializable keys to strings
        serializable_values = {str(k): v for k, v in values.items()}
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(serializable_values, f, indent=2)
            
        # Also save as numpy for convenience
        np_output_path = os.path.splitext(output_path)[0] + '.npy'
        np.save(np_output_path, np.array(list(values.values())))
        
        print(f"Values saved to {output_path} and {np_output_path}")