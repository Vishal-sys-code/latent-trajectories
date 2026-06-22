"""
ProbeResult — a structured container for trajectory analysis results.

A :class:`ProbeResult` holds the extracted trajectories, computed metrics,
and provides convenience methods for running statistical controls,
significance tests, and visualization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .trajectories import HiddenStateTrajectory

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Container returned by :meth:`GeometryProbe.run`.

    Attributes
    ----------
    trajectories : List[HiddenStateTrajectory]
        The extracted hidden-state trajectories.
    labels : Optional[List[str]]
        Semantic labels for each trajectory (may be *None*).
    metrics : Dict[str, Any]
        Pre-computed geometric metrics.  Typical keys include:
        ``trajectory_length``, ``curvature``, ``layer_velocity``,
        ``convergence_matrix``, ``rsa_matrix``, and — when labels are
        present — ``convergence_score``, ``layerwise_silhouette``.
    model_name : str
        The HuggingFace model identifier (or ``"unknown_model"``).
    """

    trajectories: List[HiddenStateTrajectory]
    labels: Optional[List[str]]
    metrics: Dict[str, Any]
    model_name: str

    # ── Controls ────────────────────────────────────────────────────────

    def controls(
        self,
        num_permutations: int = 1000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run a battery of control experiments and report pass/fail.

        Three control families are evaluated:

        1. **Label permutation** — shuffles labels and compares layerwise
           silhouette scores against a null distribution.  A control
           *passes* if the real score exceeds 95 % of the null at the
           majority of layers.
        2. **Gaussian noise** — replaces trajectories with Gaussian noise
           matched in per-layer mean/std, then compares trajectory lengths.
        3. **Temporal shuffle** — randomises layer order within each
           trajectory, then compares curvature.

        Parameters
        ----------
        num_permutations : int
            Number of label permutations (default 1 000).
        seed : int
            Random seed for reproducibility (default 42).

        Returns
        -------
        Dict[str, Any]
            A nested dictionary with keys ``"label_permutation"``,
            ``"gaussian_noise"``, and ``"temporal_shuffle"``, each
            containing a ``"passed"`` boolean and supporting statistics.
        """
        from .controls import (
            gaussian_embeddings,
            permute_labels,
            temporal_shuffle,
        )
        from .metrics import compute_curvature, compute_trajectory_length

        results: Dict[str, Any] = {}

        # ── 1.  Label permutation ───────────────────────────────────────
        if self.labels is not None and len(set(self.labels)) >= 2:
            null_distribution, real_scores = permute_labels(
                self.trajectories,
                self.labels,
                num_permutations=num_permutations,
                seed=seed,
            )

            # Per-layer p-value: fraction of null scores >= real score
            num_layers = len(real_scores)
            p_values: List[float] = []
            layers_significant = 0
            for layer_idx in range(num_layers):
                null_layer = null_distribution[:, layer_idx]
                real_val = real_scores[layer_idx]
                p = float(
                    (np.sum(null_layer >= real_val) + 1)
                    / (num_permutations + 1)
                )
                p_values.append(p)
                if p < 0.05:
                    layers_significant += 1

            passed = layers_significant > (num_layers / 2)

            results["label_permutation"] = {
                "passed": bool(passed),
                "layers_significant": layers_significant,
                "total_layers": num_layers,
                "p_values": p_values,
            }
        else:
            results["label_permutation"] = {
                "passed": None,
                "layers_significant": 0,
                "total_layers": 0,
                "p_values": [],
                "note": "Skipped — labels are missing or have fewer than 2 unique values.",
            }

        # ── 2.  Gaussian noise ──────────────────────────────────────────
        gaussian_trajs = gaussian_embeddings(self.trajectories, seed=seed)
        real_lengths = compute_trajectory_length(self.trajectories)
        null_lengths = compute_trajectory_length(gaussian_trajs)

        real_length_mean = float(np.mean(real_lengths))
        null_length_mean = float(np.mean(null_lengths))

        results["gaussian_noise"] = {
            "passed": bool(real_length_mean > null_length_mean),
            "real_length_mean": real_length_mean,
            "null_length_mean": null_length_mean,
        }

        # ── 3.  Temporal shuffle ────────────────────────────────────────
        shuffled_trajs = temporal_shuffle(self.trajectories, seed=seed)
        real_curvature = compute_curvature(self.trajectories)
        null_curvature = compute_curvature(shuffled_trajs)

        real_curvature_mean = float(np.mean(real_curvature))
        null_curvature_mean = float(np.mean(null_curvature))

        # A real trajectory should have *lower* curvature than a randomly
        # shuffled one because natural layer orderings are smooth.
        results["temporal_shuffle"] = {
            "passed": bool(real_curvature_mean < null_curvature_mean),
            "real_curvature_mean": real_curvature_mean,
            "null_curvature_mean": null_curvature_mean,
        }

        return results

    # ── Significance ────────────────────────────────────────────────────

    def significance(
        self,
        group_a_labels: Optional[List[str]] = None,
        group_b_labels: Optional[List[str]] = None,
        num_permutations: int = 10000,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compare two label groups for statistically significant differences.

        If *group_a_labels* / *group_b_labels* are not provided the method
        automatically picks the first two unique values from
        :attr:`labels`.

        Metrics compared:

        * **Trajectory length** — total path length through latent space.
        * **Convergence score** (per-prompt, mean across layers) — how
          much each trajectory's representation separates from the
          opposite group.

        Parameters
        ----------
        group_a_labels : Optional[List[str]]
            Label values that define group A (trajectories whose label is
            in this list are included).
        group_b_labels : Optional[List[str]]
            Label values that define group B.
        num_permutations : int
            Number of permutations for the permutation test (default 10 000).
        random_seed : Optional[int]
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Nested dict with ``"trajectory_length"`` and
            ``"convergence_score"`` sub-dicts, each containing
            ``p_value_permutation``, ``p_value_mwu``, and ``cohens_d``.

        Raises
        ------
        ValueError
            If labels are unavailable or fewer than 2 unique values exist.
        """
        from .metrics import (
            compute_per_prompt_convergence_score,
            compute_trajectory_length,
        )
        from .stats import compare_distributions

        if self.labels is None:
            raise ValueError(
                "Cannot run significance tests without labels."
            )

        unique_labels = sorted(set(self.labels))
        if len(unique_labels) < 2:
            raise ValueError(
                f"Need ≥ 2 unique labels for group comparison, got {unique_labels}."
            )

        # Resolve groups
        if group_a_labels is None or group_b_labels is None:
            group_a_labels = [unique_labels[0]]
            group_b_labels = [unique_labels[1]]

        group_a_set = set(group_a_labels)
        group_b_set = set(group_b_labels)

        idx_a = [i for i, l in enumerate(self.labels) if l in group_a_set]
        idx_b = [i for i, l in enumerate(self.labels) if l in group_b_set]

        if not idx_a or not idx_b:
            raise ValueError(
                "One or both groups are empty after filtering by the provided labels."
            )

        # ── Trajectory length ───────────────────────────────────────────
        all_lengths = compute_trajectory_length(self.trajectories)
        lengths_a = all_lengths[idx_a]
        lengths_b = all_lengths[idx_b]

        length_stats = compare_distributions(
            lengths_a, lengths_b,
            num_permutations=num_permutations,
            random_seed=random_seed,
        )

        # ── Convergence score (per-prompt, mean over layers) ────────────
        per_prompt_conv = compute_per_prompt_convergence_score(
            self.trajectories, self.labels,
        )  # [N, L]
        mean_conv = np.mean(per_prompt_conv, axis=1)  # [N]
        conv_a = mean_conv[idx_a]
        conv_b = mean_conv[idx_b]

        convergence_stats = compare_distributions(
            conv_a, conv_b,
            num_permutations=num_permutations,
            random_seed=random_seed,
        )

        return {
            "groups": {
                "group_a": list(group_a_set),
                "group_b": list(group_b_set),
                "n_a": len(idx_a),
                "n_b": len(idx_b),
            },
            "trajectory_length": length_stats,
            "convergence_score": convergence_stats,
        }

    # ── Visualization helpers ───────────────────────────────────────────

    def plot(
        self,
        method: str = "pca",
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs: Any,
    ) -> None:
        """Plot the trajectories in a reduced-dimensionality space.

        This is a thin wrapper around
        :func:`visualization.plot_trajectories`.

        Parameters
        ----------
        method : str
            Dimensionality-reduction method (``"pca"`` or ``"umap"``).
        save_path : Optional[str]
            If provided, the figure is saved to this path.
        show : bool
            Whether to call ``plt.show()``.
        **kwargs
            Forwarded to :func:`visualization.plot_trajectories`.
        """
        from .visualization import plot_trajectories

        plot_trajectories(
            self.trajectories,
            labels=self.labels,
            method=method,
            save_path=save_path,
            show=show,
            **kwargs,
        )

    def animate(
        self,
        save_path: str = "trajectory.gif",
        fps: int = 15,
        **kwargs: Any,
    ) -> None:
        """Create a layer-by-layer build-up animation.

        This is a thin wrapper around
        :func:`visualization.animate_trajectories`.

        Parameters
        ----------
        save_path : str
            Output file path (GIF).
        fps : int
            Frames per second.
        **kwargs
            Forwarded to :func:`visualization.animate_trajectories`.
        """
        from .visualization import animate_trajectories

        animate_trajectories(
            self.trajectories,
            labels=self.labels,
            save_path=save_path,
            fps=fps,
            **kwargs,
        )

    # ── Dunder helpers ──────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self.trajectories)
        label_info = (
            f", labels={len(set(self.labels))} unique"
            if self.labels
            else ""
        )
        metric_keys = list(self.metrics.keys())
        return (
            f"ProbeResult(model={self.model_name!r}, n_trajectories={n}"
            f"{label_info}, metrics={metric_keys})"
        )

    def __len__(self) -> int:
        return len(self.trajectories)
