# Data Valuation

This package provides implementations of various data valuation methods for machine learning.

## Description

This project implements various data valuation methods to assess the importance and contribution of individual data points in machine learning datasets. Data valuation is crucial for understanding which training examples are most valuable for model performance, identifying noisy labels, and optimizing dataset curation.

## Installation

### Requirements

- Python >= 3.11
- pip

### Setup

You can install this package directly from GitHub using pip:

```bash
pip install git+https://github.com/takumi0shibata/data-valuation.git
```

Alternatively, you can install it locally:

1. Clone the repository:
```bash
git clone https://github.com/takumi0shibata/data-valuation.git
cd data-valuation
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install the package:
```bash
pip install -e .
```

## Usage

This library provides several methods for data valuation:

1. Leave-One-Out (LOO) Valuation
2. Shapley Value-based Valuation
3. Data Valuation using Reinforcement Learning (DVRL)

Here's a basic example of how to use the library:

```python
import numpy as np
from data_valuation.core.loo_valuator import LOOValuator
from data_valuation.core.shapley_valuator import ShapleyValuator
from data_valuation.core.dvrl_valuator import DVRLValuator

# 1. Prepare your dataset
# Input data can be either:
# a) List of texts
x_train = ["text1", "text2", ...]  # Training texts
x_val = ["text3", "text4", ...]    # Validation texts
# b) Pre-computed embedding vectors as numpy arrays
x_train = np.array(...)  # Training embeddings [n_samples, embedding_dim]
x_val = np.array(...)    # Validation embeddings [n_samples, embedding_dim]

# Labels should be numpy arrays
y_train = np.array(...)  # Training labels
y_val = np.array(...)    # Validation labels

# Optional: List of unique identifiers for training samples
sample_ids = list(range(len(x_train)))  # Default: [0, 1, 2, ...]

# 2. Choose and initialize a valuation method
# Each valuator requires a prompt_id and supports various optional parameters

# Example using Leave-One-Out valuation (fastest, suitable for small datasets):
loo_valuator = LOOValuator(
    prompt_id=1,              # Required: Unique identifier for the task
    device="cuda",            # Optional: Use GPU if available
    metric="mse",            # Optional: Evaluation metric (mse/qwk/corr)
    wandb_logging=False,     # Optional: Enable Weights & Biases logging
    model_type="mlp",        # Optional: Type of model to use (mlp/features)
    batch_size=512,          # Optional: Batch size for training
    epochs=100               # Optional: Number of training epochs
)

# Example using Shapley valuation (more accurate, computationally intensive):
shapley_valuator = ShapleyValuator(
    prompt_id=1,
    device="cuda",
    metric="mse",
    max_iter=5000,          # Optional: Maximum number of iterations
    threshold=0.05          # Optional: Convergence threshold
)

# Example using DVRL (scalable to large datasets):
dvrl_valuator = DVRLValuator(
    prompt_id=1,
    device="cuda",
    metric="qwk",
    hidden_dim=100,         # Optional: Hidden dimension for value estimator
    iterations=1000,        # Optional: Number of outer iterations
    inner_iterations=100    # Optional: Number of inner iterations
)

# 3. Calculate data values
# Returns a dictionary mapping sample_ids to their computed values
values = loo_valuator.estimate_values(
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    sample_ids=sample_ids  # Optional
)

# 4. Use the values for your purpose
# For example, to identify potentially noisy samples:
noisy_samples = dict(sorted(values.items(), key=lambda x: x[1])[:10])  # 10 lowest value samples
```

### Available Valuation Methods

1. **Leave-One-Out (LOO) Valuation**
   - Simple and intuitive: measures how model performance changes when removing each sample
   - Fast for small datasets (up to a few thousand samples)
   - Provides direct measure of sample importance

2. **Shapley Valuation**
   - Based on game theory: considers all possible combinations of samples
   - Most accurate but computationally intensive
   - Recommended for small to medium datasets when theoretical guarantees are needed

3. **DVRL (Data Valuation using Reinforcement Learning)**
   - Uses reinforcement learning to learn data values efficiently
   - Scales well to large datasets
   - Best balance between accuracy and computational cost

### Common Use Cases

- Identifying noisy labels in datasets
- Finding most influential training examples
- Data cleaning and curation
- Understanding model behavior
- Detecting data quality issues
- Selecting high-value samples for active learning

For more detailed examples, please check the `examples/` directory.

## Project Structure

```
data-valuation/
├── src/
│   └── data_valuation/
│       ├── core/        # Core implementation of data valuation methods
│       ├── models/      # Model implementations
│       ├── utils/       # Utility functions
│       └── cli.py       # Command-line interface
├── tests/               # Test files
├── examples/            # Example usage
├── setup.py            # Package setup file
└── pyproject.toml      # Project configuration
```

## Dependencies

- numpy
- torch
- transformers
- scikit-learn
- wandb
- pytest
- tiktoken
- sentencepiece

## Development

To set up the development environment:

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

## Author

Takumi Shibata (shibata@ai.lab.uec.ac.jp)

## License

[License information to be added]
