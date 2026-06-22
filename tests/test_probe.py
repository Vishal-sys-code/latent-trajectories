"""
Integration tests for GeometryProbe and ProbeResult.

Uses mock models to avoid downloading real HuggingFace models in CI.
"""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from latent_trajectories import GeometryProbe, ProbeResult, HiddenStateTrajectory


# ── Mock helpers ────────────────────────────────────────────────────────


def _make_mock_model_and_tokenizer(num_layers=4, hidden_dim=8, seq_len=5):
    """Create a mock HuggingFace model and tokenizer."""
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
        result.__getitem__ = lambda self, key: torch.ones(
            1, seq_len, dtype=torch.long
        )
        return result

    tokenizer.side_effect = mock_tokenize
    tokenizer.__call__ = mock_tokenize

    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.device = torch.device("cpu")
    model.config = MagicMock()
    model.config._name_or_path = "mock-gpt2"

    # Seed for reproducibility in hidden states
    gen = torch.Generator().manual_seed(42)

    def mock_forward(**kwargs):
        result = MagicMock()
        hidden_states = tuple(
            torch.randn(1, seq_len, hidden_dim, generator=gen)
            for _ in range(num_layers + 1)
        )
        result.hidden_states = hidden_states
        return result

    model.__call__ = mock_forward
    model.side_effect = mock_forward

    return model, tokenizer


# ── GeometryProbe tests ────────────────────────────────────────────────


class TestGeometryProbeInit:
    """Test GeometryProbe initialization with different input types."""

    def test_init_with_tuple(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        probe = GeometryProbe((model, tokenizer), device="cpu")
        assert probe._model_name == "mock-gpt2"

    def test_init_with_invalid_type_raises(self):
        with pytest.raises(TypeError, match="model"):
            GeometryProbe(12345)

    def test_repr(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        probe = GeometryProbe((model, tokenizer), device="cpu")
        r = repr(probe)
        assert "mock-gpt2" in r


class TestGeometryProbeRun:
    """Test the full probe.run() pipeline."""

    def test_run_returns_probe_result(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        probe = GeometryProbe((model, tokenizer), device="cpu")
        result = probe.run(["Hello world", "Test input"])

        assert isinstance(result, ProbeResult)
        assert len(result.trajectories) == 2
        assert result.model_name == "mock-gpt2"

    def test_run_computes_core_metrics(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        probe = GeometryProbe((model, tokenizer), device="cpu")
        result = probe.run(["Hello world", "Test input"])

        # Core metrics should always be present
        assert "trajectory_length" in result.metrics
        assert "curvature" in result.metrics
        assert "layer_velocity" in result.metrics
        assert "convergence_matrix" in result.metrics
        assert "rsa_matrix" in result.metrics
        assert "summary" in result.metrics

    def test_run_with_labels_computes_label_metrics(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        probe = GeometryProbe((model, tokenizer), device="cpu")
        result = probe.run(
            ["Hello", "World", "Foo", "Bar"],
            labels=["grp_a", "grp_a", "grp_b", "grp_b"],
        )

        assert "convergence_score" in result.metrics
        assert "layerwise_silhouette" in result.metrics
        assert result.labels is not None

    def test_run_without_labels_skips_label_metrics(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        probe = GeometryProbe((model, tokenizer), device="cpu")
        result = probe.run(["Hello", "World"])

        assert "convergence_score" not in result.metrics
        assert result.labels is None

    def test_run_summary_values(self):
        model, tokenizer = _make_mock_model_and_tokenizer(
            num_layers=4, hidden_dim=8
        )
        probe = GeometryProbe((model, tokenizer), device="cpu")
        result = probe.run(["Hello"])

        summary = result.metrics["summary"]
        assert summary["n_trajectories"] == 1
        assert summary["n_layers"] == 4
        assert summary["hidden_dim"] == 8
        assert isinstance(summary["mean_trajectory_length"], float)
        assert isinstance(summary["mean_curvature"], float)


# ── ProbeResult tests ───────────────────────────────────────────────────


class TestProbeResultControls:
    """Test the controls() method."""

    def test_controls_without_labels(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        probe = GeometryProbe((model, tokenizer), device="cpu")
        result = probe.run(["Hello", "World"])

        controls = result.controls()
        assert "label_permutation" in controls
        assert "gaussian_noise" in controls
        assert "temporal_shuffle" in controls
        # Without labels, label_permutation should be skipped
        assert controls["label_permutation"]["passed"] is None

    def test_controls_with_labels(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        probe = GeometryProbe((model, tokenizer), device="cpu")
        result = probe.run(
            ["A", "B", "C", "D"],
            labels=["x", "x", "y", "y"],
        )

        controls = result.controls(num_permutations=100, seed=42)
        assert "label_permutation" in controls
        assert isinstance(controls["label_permutation"]["passed"], bool)
        assert "layers_significant" in controls["label_permutation"]
        assert "gaussian_noise" in controls
        assert isinstance(controls["gaussian_noise"]["real_length_mean"], float)
        assert "temporal_shuffle" in controls
        assert isinstance(controls["temporal_shuffle"]["real_curvature_mean"], float)


class TestProbeResultSignificance:
    """Test the significance() method."""

    def test_significance_raises_without_labels(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        probe = GeometryProbe((model, tokenizer), device="cpu")
        result = probe.run(["Hello", "World"])

        with pytest.raises(ValueError, match="labels"):
            result.significance()

    def test_significance_auto_groups(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        probe = GeometryProbe((model, tokenizer), device="cpu")
        result = probe.run(
            ["A", "B", "C", "D"],
            labels=["x", "x", "y", "y"],
        )

        sig = result.significance(num_permutations=100, random_seed=42)
        assert "groups" in sig
        assert "trajectory_length" in sig
        assert "convergence_score" in sig

    def test_significance_explicit_groups(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        probe = GeometryProbe((model, tokenizer), device="cpu")
        result = probe.run(
            ["A", "B", "C", "D"],
            labels=["x", "x", "y", "y"],
        )

        sig = result.significance(
            group_a_labels=["x"],
            group_b_labels=["y"],
            num_permutations=100,
            random_seed=42,
        )
        assert sig["groups"]["group_a"] == ["x"]
        assert sig["groups"]["group_b"] == ["y"]
        assert sig["groups"]["n_a"] == 2
        assert sig["groups"]["n_b"] == 2


class TestProbeResultDunder:
    """Test __repr__ and __len__."""

    def test_len(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        probe = GeometryProbe((model, tokenizer), device="cpu")
        result = probe.run(["A", "B", "C"])
        assert len(result) == 3

    def test_repr_contains_model_name(self):
        model, tokenizer = _make_mock_model_and_tokenizer()
        probe = GeometryProbe((model, tokenizer), device="cpu")
        result = probe.run(["A"])
        assert "mock-gpt2" in repr(result)
