"""
GeometryProbe — one-call facade for trajectory extraction and analysis.

The :class:`GeometryProbe` is the primary entry point for the
``latent_trajectories`` package.  It accepts either a HuggingFace model ID
(string) or a pre-loaded ``(model, tokenizer)`` tuple, and exposes a single
:meth:`run` method that extracts hidden states and computes all geometric
metrics in one shot.

Example
-------
>>> from latent_trajectories import GeometryProbe
>>> probe = GeometryProbe("gpt2")
>>> result = probe.run(["The cat sat on the mat.", "Explain quantum mechanics."])
>>> print(result.metrics.keys())
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .extraction import extract_hidden_states
from .metrics import (
    compute_convergence_matrix,
    compute_convergence_score,
    compute_curvature,
    compute_layer_velocity,
    compute_layerwise_silhouette,
    compute_rsa_matrix,
    compute_trajectory_length,
)
from .result import ProbeResult
from .trajectories import HiddenStateTrajectory

logger = logging.getLogger(__name__)


class GeometryProbe:
    """High-level facade for hidden-state trajectory analysis.

    Parameters
    ----------
    model : Union[str, Tuple[PreTrainedModel, PreTrainedTokenizerBase]]
        Either a HuggingFace model ID (e.g. ``"gpt2"``) that will be
        downloaded and loaded automatically, **or** a ``(model, tokenizer)``
        tuple for models already in memory.
    device : str
        Device for inference.  ``"auto"`` (default) selects CUDA when
        available, otherwise CPU.

    Examples
    --------
    Using a model ID:

    >>> probe = GeometryProbe("gpt2")

    Using a pre-loaded model:

    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> probe = GeometryProbe((model, tokenizer))
    """

    def __init__(
        self,
        model: Union[str, Tuple[PreTrainedModel, PreTrainedTokenizerBase]],
        device: str = "auto",
    ) -> None:
        self.device = device

        if isinstance(model, str):
            self._model_name = model
            logger.info("Loading model %r from HuggingFace Hub …", model)
            self._tokenizer: PreTrainedTokenizerBase = (
                AutoTokenizer.from_pretrained(model)
            )
            self._model: PreTrainedModel = (
                AutoModelForCausalLM.from_pretrained(model)
            )
        elif isinstance(model, (tuple, list)) and len(model) == 2:
            self._model, self._tokenizer = model
            self._model_name = getattr(
                self._model.config, "_name_or_path", "unknown_model"
            )
        else:
            raise TypeError(
                "'model' must be a HuggingFace model ID (str) or a "
                "(model, tokenizer) tuple.  "
                f"Got {type(model).__name__}."
            )

    # ── Public API ──────────────────────────────────────────────────────

    def run(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
    ) -> ProbeResult:
        """Extract trajectories and compute all geometric metrics.

        Parameters
        ----------
        texts : List[str]
            Input texts to analyse.
        labels : Optional[List[str]]
            Semantic labels for each text (e.g. ``"factual"`` /
            ``"reasoning"``).  When provided, label-dependent metrics
            (convergence score, silhouette) are also computed.

        Returns
        -------
        ProbeResult
            A structured result container with trajectories, metrics, and
            convenience methods for controls, significance, and plotting.

        Raises
        ------
        ValueError
            Propagated from :func:`extraction.extract_hidden_states` when
            *texts* is empty or *labels* length mismatches.
        """
        # ── Extraction ──────────────────────────────────────────────────
        trajectories = extract_hidden_states(
            model=self._model,
            tokenizer=self._tokenizer,
            texts=texts,
            labels=labels,
            device=self.device,
        )

        # ── Metrics ─────────────────────────────────────────────────────
        metrics: Dict[str, Any] = {}

        metrics["trajectory_length"] = compute_trajectory_length(trajectories)
        metrics["trajectory_length_unnormalized"] = compute_trajectory_length(
            trajectories, normalized=False
        )
        metrics["curvature"] = compute_curvature(trajectories)
        metrics["layer_velocity"] = compute_layer_velocity(trajectories)
        metrics["convergence_matrix"] = compute_convergence_matrix(trajectories)
        metrics["rsa_matrix"] = compute_rsa_matrix(trajectories)

        # Label-dependent metrics
        if labels is not None and len(set(labels)) >= 2:
            metrics["convergence_score"] = compute_convergence_score(
                trajectories, labels
            )
            metrics["layerwise_silhouette"] = compute_layerwise_silhouette(
                trajectories, labels
            )
        else:
            logger.debug(
                "Skipping label-dependent metrics (labels=%s).",
                "None" if labels is None else f"{len(set(labels))} unique",
            )

        # ── Summary statistics ──────────────────────────────────────────
        metrics["summary"] = {
            "n_trajectories": len(trajectories),
            "n_layers": trajectories[0].num_layers if trajectories else 0,
            "hidden_dim": trajectories[0].hidden_dim if trajectories else 0,
            "mean_trajectory_length": float(
                np.mean(metrics["trajectory_length"])
            ),
            "mean_curvature": float(np.mean(metrics["curvature"])),
        }

        return ProbeResult(
            trajectories=trajectories,
            labels=labels,
            metrics=metrics,
            model_name=self._model_name,
        )

    # ── Dunder helpers ──────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GeometryProbe(model={self._model_name!r}, device={self.device!r})"
        )
