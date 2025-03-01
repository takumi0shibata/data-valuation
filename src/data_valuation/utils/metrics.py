import numpy as np
from sklearn.metrics import mean_squared_error, cohen_kappa_score
from scipy.stats import pearsonr
from typing import Union, Literal

from ..utils.data_utils import get_min_max_scores


def calculate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prompt_id: int,
    metric: Literal["mse", "corr", "qwk"] = "mse"
) -> float:
    """
    Calculate the specified metric between true and predicted values.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metric: Metric to calculate. One of:
            - "mse": Mean Squared Error
            - "corr": Pearson correlation coefficient
            - "qwk": Quadratic Weighted Kappa
            
    Returns:
        float: Calculated metric value
        
    Raises:
        ValueError: If an invalid metric is specified
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Flatten arrays if they are 2D
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
        
    if metric == "mse":
        return mean_squared_error(y_true, y_pred)
    
    elif metric == "corr":
        correlation, _ = pearsonr(y_true, y_pred)
        return correlation
    
    elif metric == "qwk":
        # For QWK, we need integer labels and rescale based on prompt_id
        minscore, maxscore = get_min_max_scores()[prompt_id]['score']
        y_true = np.round((maxscore - minscore) * np.array(y_true) + minscore)
        y_pred = np.round((maxscore - minscore) * np.array(y_pred) + minscore)
        return cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[i for i in range(minscore, maxscore + 1)])
    
    else:
        raise ValueError(f"Unknown metric: {metric}. Must be one of: mse, corr, qwk")
