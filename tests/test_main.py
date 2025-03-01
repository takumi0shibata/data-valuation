import pytest
import numpy as np
import torch
from typing import List, Dict, Any

from src.data_valuation.core import (
    ShapleyValuator,
    LOOValuator,
    DVRLValuator
)

@pytest.fixture
def numeric_data():
    """Generate synthetic numeric data for testing."""
    np.random.seed(42)
    x_train = np.random.randn(10, 5)
    y_train = np.random.randn(10, 1)
    x_val = np.random.randn(5, 5)
    y_val = np.random.randn(5, 1)
    return x_train, y_train, x_val, y_val

@pytest.fixture
def text_data():
    """Generate synthetic text data for testing."""
    x_train = [
        "This is a positive example.",
        "Another good example.",
        "This is a negative example.",
        "Yet another negative case.",
        "A neutral example."
    ]
    y_train = np.array([1, 1, 0, 0, 0.5]).reshape(-1, 1)
    
    x_val = [
        "A new positive case.",
        "A new negative case.",
        "Another neutral example."
    ]
    y_val = np.array([1, 0, 0.5]).reshape(-1, 1)
    
    return x_train, y_train, x_val, y_val

@pytest.fixture
def sample_ids():
    """Generate sample IDs for testing."""
    return ["id_" + str(i) for i in range(10)]

class TestShapleyValuator:
    def test_initialization(self):
        """Test ShapleyValuator initialization."""
        valuator = ShapleyValuator(
            prompt_id=1,
            device="cpu",
            metric="mse",
            max_iter=100
        )
        assert valuator.prompt_id == 1
        assert valuator.device == "cpu"
        assert valuator.metric == "mse"
        assert valuator.max_iter == 100
        
    def test_numeric_data(self, numeric_data):
        """Test ShapleyValuator with numeric data."""
        x_train, y_train, x_val, y_val = numeric_data
        valuator = ShapleyValuator(prompt_id=1, max_iter=10)  # Small max_iter for testing
        
        values = valuator.estimate_values(x_train, y_train, x_val, y_val)
        
        assert isinstance(values, dict)
        assert len(values) == len(x_train)
        assert all(isinstance(v, float) for v in values.values())
        
    def test_text_data(self, text_data):
        """Test ShapleyValuator with text data."""
        x_train, y_train, x_val, y_val = text_data
        valuator = ShapleyValuator(prompt_id=1, max_iter=10)  # Small max_iter for testing
        
        values = valuator.estimate_values(x_train, y_train, x_val, y_val)
        
        assert isinstance(values, dict)
        assert len(values) == len(x_train)
        assert all(isinstance(v, float) for v in values.values())
        
    def test_with_sample_ids(self, numeric_data, sample_ids):
        """Test ShapleyValuator with custom sample IDs."""
        x_train, y_train, x_val, y_val = numeric_data
        valuator = ShapleyValuator(prompt_id=1, max_iter=10)
        
        values = valuator.estimate_values(
            x_train[:len(sample_ids)],
            y_train[:len(sample_ids)],
            x_val,
            y_val,
            sample_ids
        )
        
        assert set(values.keys()) == set(sample_ids)
        
    def test_invalid_input(self, numeric_data):
        """Test ShapleyValuator with invalid inputs."""
        x_train, y_train, x_val, y_val = numeric_data
        valuator = ShapleyValuator(prompt_id=1)
        
        # Test mismatched sample_ids
        with pytest.raises(ValueError):
            valuator.estimate_values(
                x_train,
                y_train,
                x_val,
                y_val,
                sample_ids=["id_1"]  # Wrong length
            )

class TestLOOValuator:
    def test_initialization(self):
        """Test LOOValuator initialization."""
        valuator = LOOValuator(
            prompt_id=1,
            device="cpu",
            metric="mse",
            batch_size=32,
            epochs=10
        )
        assert valuator.prompt_id == 1
        assert valuator.device == "cpu"
        assert valuator.metric == "mse"
        assert valuator.batch_size == 32
        assert valuator.epochs == 10
        
    def test_numeric_data(self, numeric_data):
        """Test LOOValuator with numeric data."""
        x_train, y_train, x_val, y_val = numeric_data
        valuator = LOOValuator(prompt_id=1, epochs=2)  # Small epochs for testing
        
        values = valuator.estimate_values(x_train, y_train, x_val, y_val)
        
        assert isinstance(values, dict)
        assert len(values) == len(x_train)
        assert all(isinstance(v, float) for v in values.values())
        
    def test_text_data(self, text_data):
        """Test LOOValuator with text data."""
        x_train, y_train, x_val, y_val = text_data
        valuator = LOOValuator(prompt_id=1, epochs=2)  # Small epochs for testing
        
        values = valuator.estimate_values(x_train, y_train, x_val, y_val)
        
        assert isinstance(values, dict)
        assert len(values) == len(x_train)
        assert all(isinstance(v, float) for v in values.values())
        
    def test_with_sample_ids(self, numeric_data, sample_ids):
        """Test LOOValuator with custom sample IDs."""
        x_train, y_train, x_val, y_val = numeric_data
        valuator = LOOValuator(prompt_id=1, epochs=2)
        
        values = valuator.estimate_values(
            x_train[:len(sample_ids)],
            y_train[:len(sample_ids)],
            x_val,
            y_val,
            sample_ids
        )
        
        assert set(values.keys()) == set(sample_ids)
        
    def test_invalid_input(self, numeric_data):
        """Test LOOValuator with invalid inputs."""
        x_train, y_train, x_val, y_val = numeric_data
        valuator = LOOValuator(prompt_id=1)
        
        # Test mismatched sample_ids
        with pytest.raises(ValueError):
            valuator.estimate_values(
                x_train,
                y_train,
                x_val,
                y_val,
                sample_ids=["id_1"]  # Wrong length
            )

class TestDVRLValuator:
    def test_initialization(self):
        """Test DVRLValuator initialization."""
        valuator = DVRLValuator(
            prompt_id=1,
            device="cpu",
            metric="mse",
            iterations=100,
            batch_size=32
        )
        assert valuator.prompt_id == 1
        assert valuator.device == "cpu"
        assert valuator.metric == "mse"
        assert valuator.iterations == 100
        assert valuator.batch_size == 32
        
    def test_numeric_data(self, numeric_data):
        """Test DVRLValuator with numeric data."""
        x_train, y_train, x_val, y_val = numeric_data
        valuator = DVRLValuator(prompt_id=1, iterations=2)  # Small iterations for testing
        
        values = valuator.estimate_values(x_train, y_train, x_val, y_val)
        
        assert isinstance(values, dict)
        assert len(values) == len(x_train)
        assert all(isinstance(v, float) for v in values.values())
        
    def test_text_data(self, text_data):
        """Test DVRLValuator with text data."""
        x_train, y_train, x_val, y_val = text_data
        valuator = DVRLValuator(prompt_id=1, iterations=2)  # Small iterations for testing
        
        values = valuator.estimate_values(x_train, y_train, x_val, y_val)
        
        assert isinstance(values, dict)
        assert len(values) == len(x_train)
        assert all(isinstance(v, float) for v in values.values())
        
    def test_with_sample_ids(self, numeric_data, sample_ids):
        """Test DVRLValuator with custom sample IDs."""
        x_train, y_train, x_val, y_val = numeric_data
        valuator = DVRLValuator(prompt_id=1, iterations=2)
        
        values = valuator.estimate_values(
            x_train[:len(sample_ids)],
            y_train[:len(sample_ids)],
            x_val,
            y_val,
            sample_ids
        )
        
        assert set(values.keys()) == set(sample_ids)
        
    def test_invalid_input(self, numeric_data):
        """Test DVRLValuator with invalid inputs."""
        x_train, y_train, x_val, y_val = numeric_data
        valuator = DVRLValuator(prompt_id=1)
        
        # Test mismatched sample_ids
        with pytest.raises(ValueError):
            valuator.estimate_values(
                x_train,
                y_train,
                x_val,
                y_val,
                sample_ids=["id_1"]  # Wrong length
            )
        
    def test_1d_to_2d_conversion(self, numeric_data):
        """Test DVRLValuator's handling of 1D labels."""
        x_train, y_train, x_val, y_val = numeric_data
        valuator = DVRLValuator(prompt_id=1, iterations=2)
        
        # Convert labels to 1D
        y_train_1d = y_train.squeeze()
        y_val_1d = y_val.squeeze()
        
        values = valuator.estimate_values(x_train, y_train_1d, x_val, y_val_1d)
        
        assert isinstance(values, dict)
        assert len(values) == len(x_train)
        assert all(isinstance(v, float) for v in values.values())
