"""
Visualization utilities for hidden-state trajectories.

Provides two public functions:

* :func:`plot_trajectories` — static 3-D scatter / line plot of
  dimensionality-reduced trajectories, coloured by label.
* :func:`animate_trajectories` — layer-by-layer build-up GIF animation.

Both functions use PCA (or optionally UMAP) to project the
high-dimensional hidden states into 3-D space and rely on *matplotlib*
with a clean dark theme and the *seaborn* colour palette.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import animation
from sklearn.decomposition import PCA

from .trajectories import HiddenStateTrajectory

logger = logging.getLogger(__name__)

# ── Theme ───────────────────────────────────────────────────────────────

_DARK_STYLE: Dict[str, Any] = {
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#1e1e2e",
    "axes.edgecolor": "#585b70",
    "axes.labelcolor": "#cdd6f4",
    "text.color": "#cdd6f4",
    "xtick.color": "#a6adc8",
    "ytick.color": "#a6adc8",
    "grid.color": "#45475a",
    "grid.alpha": 0.4,
    "legend.facecolor": "#313244",
    "legend.edgecolor": "#585b70",
}


def _flatten_trajectories(
    trajectories: List[HiddenStateTrajectory],
) -> np.ndarray:
    """Stack all trajectory tensors into a single ``[N*L, D]`` matrix."""
    return np.concatenate(
        [t.trajectory.numpy() for t in trajectories], axis=0
    )


def _build_projector(
    flat_data: np.ndarray,
    method: str = "pca",
    n_components: int = 3,
) -> np.ndarray:
    """Fit a dimensionality-reduction model and return projected data.

    Parameters
    ----------
    flat_data : np.ndarray
        Shape ``[N*L, D]``.
    method : str
        ``"pca"`` (default) or ``"umap"``.
    n_components : int
        Number of output dimensions.

    Returns
    -------
    np.ndarray
        Shape ``[N*L, n_components]``.
    """
    if method == "pca":
        effective_components = min(n_components, flat_data.shape[0], flat_data.shape[1])
        model = PCA(n_components=effective_components)
        projected = model.fit_transform(flat_data)
    elif method == "umap":
        try:
            import umap  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "umap-learn is required for method='umap'.  "
                "Install it with:  pip install umap-learn"
            ) from exc
        n_neighbors = min(15, flat_data.shape[0] - 1)
        if n_neighbors < 2:
            n_neighbors = 2
        model = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=n_neighbors)
        projected = model.fit_transform(flat_data)
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'pca' or 'umap'.")

    # Pad if effective components < n_components
    if projected.shape[1] < n_components:
        padding = np.zeros((projected.shape[0], n_components - projected.shape[1]))
        projected = np.concatenate([projected, padding], axis=1)

    return projected


def _label_to_colour_map(
    labels: Optional[List[str]],
    n_trajectories: int,
) -> tuple:
    """Build a colour array and a label→colour mapping.

    Returns
    -------
    colours : list
        One colour per trajectory.
    colour_map : dict
        ``{label: colour}`` for legend construction.
    """
    palette = sns.color_palette("husl", n_colors=max(8, n_trajectories))

    if labels is None:
        colours = [palette[i % len(palette)] for i in range(n_trajectories)]
        return colours, {}

    unique_labels = sorted(set(labels))
    label_palette = sns.color_palette("husl", n_colors=len(unique_labels))
    colour_map = {lbl: label_palette[i] for i, lbl in enumerate(unique_labels)}
    colours = [colour_map[l] for l in labels]
    return colours, colour_map


# ── Public API ──────────────────────────────────────────────────────────


def plot_trajectories(
    trajectories: List[HiddenStateTrajectory],
    labels: Optional[List[str]] = None,
    method: str = "pca",
    n_components: int = 3,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (12, 8),
) -> None:
    """Plot trajectories as coloured 3-D lines in a reduced space.

    Each trajectory is drawn as a line connecting successive layers,
    with markers at each layer position.  If *labels* are provided the
    trajectories are coloured by group and a legend is shown.

    Parameters
    ----------
    trajectories : List[HiddenStateTrajectory]
        Trajectories to visualise.
    labels : Optional[List[str]]
        One label per trajectory for colouring.
    method : str
        ``"pca"`` (default) or ``"umap"``.
    n_components : int
        Number of projection dimensions (fixed at 3 for the 3-D plot).
    save_path : Optional[str]
        If provided, the figure is saved to this path (PNG, PDF, …).
    show : bool
        Whether to call ``plt.show()``.
    figsize : tuple
        Figure size in inches.
    """
    if not trajectories:
        logger.warning("No trajectories to plot.")
        return

    n = len(trajectories)
    num_layers = trajectories[0].num_layers

    # Project
    flat = _flatten_trajectories(trajectories)
    projected = _build_projector(flat, method=method, n_components=3)
    projected_3d = projected.reshape(n, num_layers, 3)

    # Colours
    colours, colour_map = _label_to_colour_map(labels, n)

    with plt.rc_context(_DARK_STYLE):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(_DARK_STYLE["axes.facecolor"])

        # Draw trajectories
        for i in range(n):
            xs = projected_3d[i, :, 0]
            ys = projected_3d[i, :, 1]
            zs = projected_3d[i, :, 2]
            lbl = labels[i] if labels else None
            # Only add label for legend on the first occurrence of each group
            show_label = (
                lbl
                if (lbl and (i == 0 or labels[i] != labels[i - 1] or i == next(
                    j for j, l in enumerate(labels) if l == lbl
                )))
                else None
            )
            ax.plot(
                xs, ys, zs,
                color=colours[i],
                alpha=0.7,
                linewidth=1.2,
                label=show_label,
            )
            # Start marker
            ax.scatter(
                xs[0], ys[0], zs[0],
                color=colours[i], s=40, marker="o", edgecolors="white",
                linewidths=0.5, zorder=5,
            )
            # End marker
            ax.scatter(
                xs[-1], ys[-1], zs[-1],
                color=colours[i], s=60, marker="*", edgecolors="white",
                linewidths=0.5, zorder=5,
            )

        method_upper = method.upper()
        ax.set_xlabel(f"{method_upper} 1", fontsize=10)
        ax.set_ylabel(f"{method_upper} 2", fontsize=10)
        ax.set_zlabel(f"{method_upper} 3", fontsize=10)
        ax.set_title(
            f"Hidden-State Trajectories ({method_upper})",
            fontsize=14, fontweight="bold", pad=15,
        )

        if colour_map:
            ax.legend(
                loc="upper left", fontsize=9, framealpha=0.8,
                markerscale=0.8,
            )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Figure saved to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)


def animate_trajectories(
    trajectories: List[HiddenStateTrajectory],
    labels: Optional[List[str]] = None,
    method: str = "pca",
    save_path: str = "trajectory.gif",
    fps: int = 15,
    figsize: tuple = (12, 8),
) -> None:
    """Create a layer-by-layer build-up animation saved as a GIF.

    Each frame adds the next layer's points and line segments for all
    trajectories, progressively revealing the full geometry.

    Parameters
    ----------
    trajectories : List[HiddenStateTrajectory]
        Trajectories to animate.
    labels : Optional[List[str]]
        One label per trajectory for colouring.
    method : str
        ``"pca"`` (default) or ``"umap"``.
    save_path : str
        Output GIF file path (default ``"trajectory.gif"``).
    fps : int
        Frames per second (default 15).
    figsize : tuple
        Figure size in inches.
    """
    if not trajectories:
        logger.warning("No trajectories to animate.")
        return

    n = len(trajectories)
    num_layers = trajectories[0].num_layers

    # Project
    flat = _flatten_trajectories(trajectories)
    projected = _build_projector(flat, method=method, n_components=3)
    projected_3d = projected.reshape(n, num_layers, 3)

    # Colours
    colours, colour_map = _label_to_colour_map(labels, n)

    # Axis limits with some padding
    all_coords = projected_3d.reshape(-1, 3)
    mins = all_coords.min(axis=0)
    maxs = all_coords.max(axis=0)
    pad = (maxs - mins) * 0.1
    mins -= pad
    maxs += pad

    with plt.rc_context(_DARK_STYLE):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(_DARK_STYLE["axes.facecolor"])

        method_upper = method.upper()
        ax.set_xlabel(f"{method_upper} 1", fontsize=10)
        ax.set_ylabel(f"{method_upper} 2", fontsize=10)
        ax.set_zlabel(f"{method_upper} 3", fontsize=10)

        def _init():
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
            return []

        def _update(frame: int):
            ax.cla()
            ax.set_facecolor(_DARK_STYLE["axes.facecolor"])
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
            ax.set_xlabel(f"{method_upper} 1", fontsize=10)
            ax.set_ylabel(f"{method_upper} 2", fontsize=10)
            ax.set_zlabel(f"{method_upper} 3", fontsize=10)

            layer_idx = frame + 1  # how many layers to show (1-indexed)
            ax.set_title(
                f"Layer {frame} / {num_layers - 1}",
                fontsize=14, fontweight="bold", pad=15,
            )

            legend_seen: set = set()
            for i in range(n):
                xs = projected_3d[i, :layer_idx, 0]
                ys = projected_3d[i, :layer_idx, 1]
                zs = projected_3d[i, :layer_idx, 2]

                lbl = labels[i] if labels else None
                show_label = None
                if lbl and lbl not in legend_seen:
                    show_label = lbl
                    legend_seen.add(lbl)

                ax.plot(
                    xs, ys, zs,
                    color=colours[i], alpha=0.7, linewidth=1.2,
                    label=show_label,
                )
                # Current head marker
                ax.scatter(
                    xs[-1], ys[-1], zs[-1],
                    color=colours[i], s=30, marker="o",
                    edgecolors="white", linewidths=0.5, zorder=5,
                )

            if colour_map:
                ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

            return []

        anim = animation.FuncAnimation(
            fig,
            _update,
            init_func=_init,
            frames=num_layers,
            interval=1000 // fps,
            blit=False,
        )

        anim.save(save_path, writer="pillow", fps=fps)
        logger.info("Animation saved to %s", save_path)
        plt.close(fig)
