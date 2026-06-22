"""
Generalized hidden-state extraction for any HuggingFace causal language model.

This module provides a single public function, :func:`extract_hidden_states`,
that takes a pre-loaded model + tokenizer pair (or just the model when the
tokenizer is bundled) and a list of text strings, runs each text through the
model, and returns a list of :class:`HiddenStateTrajectory` objects ready for
geometric analysis.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .trajectories import HiddenStateTrajectory

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> torch.device:
    """Resolve the ``"auto"`` device string to a concrete :class:`torch.device`.

    Parameters
    ----------
    device : str
        ``"auto"`` selects CUDA when available, otherwise CPU.  Any other
        string is passed through to :class:`torch.device`.

    Returns
    -------
    torch.device
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def extract_hidden_states(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    labels: Optional[List[str]] = None,
    device: str = "auto",
) -> List[HiddenStateTrajectory]:
    """Extract hidden-state trajectories from a causal language model.

    For every input text the function:

    1. Tokenizes the text (no padding, single-sequence).
    2. Runs a forward pass with ``output_hidden_states=True``.
    3. Stacks all hidden states into a ``[num_layers+1, seq_len, hidden_dim]``
       tensor, then selects the **last token** position.
    4. Splits the result into an *embedding state* (layer 0) and a
       *trajectory* (layers 1 … N).
    5. Wraps everything in a :class:`HiddenStateTrajectory`.

    Parameters
    ----------
    model : PreTrainedModel
        A HuggingFace causal language model (e.g. ``AutoModelForCausalLM``).
        The model is placed into ``eval()`` mode and wrapped in
        ``torch.no_grad()`` automatically.
    tokenizer : PreTrainedTokenizerBase
        The matching tokenizer.  If it has no ``pad_token`` set, the
        ``eos_token`` is used as a fallback.
    texts : List[str]
        Input texts to analyse.  Each text is processed independently.
    labels : Optional[List[str]]
        Optional semantic labels corresponding to each text.  These are stored
        as the ``model_family`` field of the resulting trajectories for
        convenient grouping downstream.  When *None*, ``model_family`` defaults
        to ``"unknown"``.
    device : str
        Device for inference.  ``"auto"`` (default) selects CUDA when
        available, otherwise CPU.

    Returns
    -------
    List[HiddenStateTrajectory]
        One trajectory per input text, with ``embedding_state`` (shape
        ``[D]``) and ``trajectory`` (shape ``[L, D]``) stored as
        ``float32`` tensors on CPU.

    Raises
    ------
    ValueError
        If *texts* is empty or if *labels* is provided but its length does
        not match the length of *texts*.
    """
    # ── Validation ──────────────────────────────────────────────────────
    if not texts:
        raise ValueError("'texts' must be a non-empty list of strings.")

    if labels is not None and len(labels) != len(texts):
        raise ValueError(
            f"Length mismatch: got {len(texts)} texts but {len(labels)} labels."
        )

    # ── Device & model setup ────────────────────────────────────────────
    resolved_device = _resolve_device(device)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(resolved_device)
    model.eval()

    # Infer model name from the config (best-effort)
    model_name: str = getattr(model.config, "_name_or_path", "unknown_model")

    # ── Extract trajectories ────────────────────────────────────────────
    trajectories: List[HiddenStateTrajectory] = []

    with torch.no_grad():
        for idx, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(resolved_device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)

            # outputs.hidden_states is a tuple of length (num_layers + 1).
            # Each element: [batch=1, seq_len, hidden_dim].
            # Stack → [num_layers+1, seq_len, hidden_dim], squeeze batch.
            stacked = (
                torch.stack(outputs.hidden_states)
                .squeeze(1)        # remove batch dim
                .float()           # ensure float32
                .cpu()
            )

            # Select the last token across all layers → [num_layers+1, D]
            last_token_states = stacked[:, -1, :]

            # Split: layer 0 = embedding, layers 1..N = transformer blocks
            embedding_state = last_token_states[0]        # [D]
            trajectory = last_token_states[1:]             # [L, D]

            label = labels[idx] if labels is not None else "unknown"

            traj = HiddenStateTrajectory(
                prompt_id=idx,
                prompt=text,
                model=model_name,
                embedding_state=embedding_state,
                trajectory=trajectory,
                model_family=label,
            )
            trajectories.append(traj)

            logger.debug(
                "Extracted trajectory %d/%d  (layers=%d, dim=%d)",
                idx + 1,
                len(texts),
                traj.num_layers,
                traj.hidden_dim,
            )

    return trajectories
