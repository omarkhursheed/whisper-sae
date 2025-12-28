"""Pytest configuration and shared fixtures."""

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture(scope="session")
def cpu_device():
    """Always return CPU device for consistent testing."""
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
