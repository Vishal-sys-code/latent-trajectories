"""
Latent Trajectories — Geometric analysis of transformer hidden-state trajectories.

This package provides tools for extracting, analysing, and visualising the
geometric structure of hidden-state trajectories across transformer layers.

Quick start::

    from latent_trajectories import GeometryProbe

    probe = GeometryProbe("gpt2")
    result = probe.run(
        texts=["The cat sat on the mat.", "Explain quantum mechanics."],
        labels=["factual", "reasoning"],
    )

    print(result.metrics["summary"])
    result.controls()       # statistical controls
    result.significance()   # Mann-Whitney / permutation tests
    result.plot()           # 3-D trajectory visualisation
"""

from latent_trajectories.probe import GeometryProbe
from latent_trajectories.result import ProbeResult
from latent_trajectories.trajectories import HiddenStateTrajectory
from latent_trajectories import load_prompts

__version__ = "0.1.0"

__all__ = [
    "GeometryProbe",
    "ProbeResult",
    "HiddenStateTrajectory",
    "load_prompts",
    "__version__",
]
