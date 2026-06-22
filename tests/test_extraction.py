"""Tests for the generalized extraction module."""
import pytest
import torch
from unittest.mock import MagicMock, patch
from latent_trajectories.extraction import extract_hidden_states
from latent_trajectories.trajectories import HiddenStateTrajectory


def _make_mock_model_and_tokenizer(num_layers=4, hidden_dim=8, seq_len=5):
    """Create mock model and tokenizer for testing extraction without loading real models."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"
    
    def mock_tokenize(text, return_tensors=None):
        result = MagicMock()
        result.__iter__ = lambda self: iter(["input_ids", "attention_mask"])
        result.items = lambda: [
            ("input_ids", torch.ones(1, seq_len, dtype=torch.long)),
            ("attention_mask", torch.ones(1, seq_len, dtype=torch.long)),
        ]
        result.__getitem__ = lambda self, key: torch.ones(1, seq_len, dtype=torch.long)
        return result
    
    tokenizer.side_effect = mock_tokenize
    tokenizer.__call__ = mock_tokenize
    
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.device = torch.device("cpu")
    
    def mock_forward(**kwargs):
        result = MagicMock()
        # Create hidden states: tuple of (num_layers+1) tensors, each [batch, seq_len, hidden_dim]
        hidden_states = tuple(
            torch.randn(1, seq_len, hidden_dim) for _ in range(num_layers + 1)
        )
        result.hidden_states = hidden_states
        return result
    
    model.__call__ = mock_forward
    model.side_effect = mock_forward
    
    return model, tokenizer


def test_extract_returns_trajectories():
    model, tokenizer = _make_mock_model_and_tokenizer(num_layers=4, hidden_dim=8)
    texts = ["Hello world", "Test input"]
    trajectories = extract_hidden_states(model, tokenizer, texts, device="cpu")
    
    assert len(trajectories) == 2
    assert all(isinstance(t, HiddenStateTrajectory) for t in trajectories)


def test_extract_trajectory_shape():
    model, tokenizer = _make_mock_model_and_tokenizer(num_layers=4, hidden_dim=8)
    texts = ["Hello world"]
    trajectories = extract_hidden_states(model, tokenizer, texts, device="cpu")
    
    traj = trajectories[0]
    assert traj.num_layers == 4  # num_layers transformer blocks (excluding embedding)
    assert traj.hidden_dim == 8


def test_extract_with_labels():
    model, tokenizer = _make_mock_model_and_tokenizer(num_layers=4, hidden_dim=8)
    texts = ["Hello", "World"]
    labels = ["greetings", "nouns"]
    trajectories = extract_hidden_states(model, tokenizer, texts, labels=labels, device="cpu")
    
    assert trajectories[0].model_family == "greetings"
    assert trajectories[1].model_family == "nouns"


def test_extract_empty_texts():
    model, tokenizer = _make_mock_model_and_tokenizer()
    with pytest.raises(ValueError, match="non-empty"):
        extract_hidden_states(model, tokenizer, [], device="cpu")

