"""
Command-line interface for the ``latent-trajectories`` package.

Entry point registered as ``latent-trajectories`` in ``pyproject.toml``::

    latent-trajectories analyze gpt2 --text "Hello world" --plot
    latent-trajectories info
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional


def _resolve_texts_and_labels(
    args: argparse.Namespace,
) -> tuple:
    """Return ``(texts, labels)`` from the parsed CLI arguments.

    Texts come from either ``--text`` (inline) or ``--texts`` (file).
    Labels come from ``--labels`` (file) and are optional.

    Returns
    -------
    tuple of (List[str], Optional[List[str]])
    """
    texts: List[str] = []

    if args.text:
        texts = args.text
    elif args.texts:
        if not os.path.isfile(args.texts):
            print(f"Error: texts file not found: {args.texts}", file=sys.stderr)
            sys.exit(1)
        with open(args.texts, "r", encoding="utf-8") as fh:
            texts = [line.strip() for line in fh if line.strip()]
    else:
        print(
            "Error: provide texts via --text or --texts.",
            file=sys.stderr,
        )
        sys.exit(1)

    labels: Optional[List[str]] = None
    if args.labels:
        if not os.path.isfile(args.labels):
            print(f"Error: labels file not found: {args.labels}", file=sys.stderr)
            sys.exit(1)
        with open(args.labels, "r", encoding="utf-8") as fh:
            labels = [line.strip() for line in fh if line.strip()]
        if len(labels) != len(texts):
            print(
                f"Error: {len(labels)} labels but {len(texts)} texts.",
                file=sys.stderr,
            )
            sys.exit(1)

    return texts, labels


def _cmd_analyze(args: argparse.Namespace) -> None:
    """Execute the ``analyze`` sub-command."""
    # Lazy imports so the CLI parser itself stays fast
    from .probe import GeometryProbe

    texts, labels = _resolve_texts_and_labels(args)

    print(f"Loading model: {args.model}")
    probe = GeometryProbe(args.model, device=args.device)

    print(f"Analysing {len(texts)} text(s) …")
    result = probe.run(texts, labels=labels)

    # ── Print summary ───────────────────────────────────────────────────
    summary = result.metrics.get("summary", {})
    print("\n── Summary ──────────────────────────────────────────")
    print(f"  Model           : {result.model_name}")
    print(f"  Trajectories    : {summary.get('n_trajectories', '?')}")
    print(f"  Layers          : {summary.get('n_layers', '?')}")
    print(f"  Hidden dim      : {summary.get('hidden_dim', '?')}")
    print(f"  Mean traj length: {summary.get('mean_trajectory_length', '?'):.4f}")
    print(f"  Mean curvature  : {summary.get('mean_curvature', '?'):.4f}")

    # ── Controls ────────────────────────────────────────────────────────
    if args.controls:
        print("\n── Controls ─────────────────────────────────────────")
        ctrl = result.controls()
        for name, info in ctrl.items():
            passed = info.get("passed")
            status = (
                "PASS" if passed is True
                else "FAIL" if passed is False
                else "SKIP"
            )
            print(f"  [{status}]  {name}")
            for k, v in info.items():
                if k in ("passed", "p_values"):
                    continue
                print(f"         {k}: {v}")

    # ── Save metrics ────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.json")

    # Serialise metrics (convert numpy arrays to lists)
    serialisable: dict = {}
    for key, val in result.metrics.items():
        try:
            import numpy as np

            if isinstance(val, np.ndarray):
                serialisable[key] = val.tolist()
            elif isinstance(val, dict):
                serialisable[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in val.items()
                }
            else:
                serialisable[key] = val
        except Exception:
            serialisable[key] = str(val)

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, default=str)
    print(f"\nMetrics saved to {metrics_path}")

    # ── Plot ────────────────────────────────────────────────────────────
    if args.plot:
        plot_path = os.path.join(args.output_dir, "trajectories.png")
        result.plot(save_path=plot_path, show=False)
        print(f"Plot saved to {plot_path}")

    # ── Animate ─────────────────────────────────────────────────────────
    if args.animate:
        anim_path = os.path.join(args.output_dir, "trajectories.gif")
        result.animate(save_path=anim_path)
        print(f"Animation saved to {anim_path}")


def _cmd_info(args: argparse.Namespace) -> None:
    """Execute the ``info`` sub-command."""
    from . import __version__

    print(f"latent-trajectories  v{__version__}")
    print("Geometric analysis of transformer hidden-state trajectories")
    print()
    print("Modules:")
    print("  extraction        — hidden-state extraction")
    print("  metrics           — trajectory geometry (length, curvature, RSA, …)")
    print("  controls          — label permutation, Gaussian noise, temporal shuffle")
    print("  stats             — bootstrap CI, permutation test, Cohen's d")
    print("  visualization     — 3-D plots and GIF animations")
    print("  probe             — GeometryProbe (high-level API)")
    print("  result            — ProbeResult container")
    print()
    print("Usage:")
    print("  latent-trajectories analyze gpt2 --text 'Hello world' --plot")
    print("  latent-trajectories info")


def main() -> None:
    """CLI entry point (registered in ``pyproject.toml`` as ``latent-trajectories``)."""
    parser = argparse.ArgumentParser(
        prog="latent-trajectories",
        description="Geometric analysis of transformer hidden-state trajectories",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── analyze ─────────────────────────────────────────────────────────
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run full trajectory analysis on a HuggingFace model",
    )
    analyze_parser.add_argument(
        "model",
        help="HuggingFace model ID (e.g., gpt2, TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    analyze_parser.add_argument(
        "--texts",
        help="Path to a text file with one input text per line",
    )
    analyze_parser.add_argument(
        "--text",
        nargs="+",
        help="Inline text(s) to analyse (space-separated strings)",
    )
    analyze_parser.add_argument(
        "--labels",
        help="Path to a labels file (one label per line, matching --texts order)",
    )
    analyze_parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output files (default: current directory)",
    )
    analyze_parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference: 'auto', 'cpu', 'cuda', 'cuda:0', … (default: auto)",
    )
    analyze_parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a 3-D trajectory plot (PNG)",
    )
    analyze_parser.add_argument(
        "--animate",
        action="store_true",
        help="Generate a layer-by-layer trajectory animation (GIF)",
    )
    analyze_parser.add_argument(
        "--controls",
        action="store_true",
        help="Run control experiments (label permutation, Gaussian noise, temporal shuffle)",
    )

    # ── info ────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "info",
        help="Show package version and available modules",
    )

    args = parser.parse_args()

    if args.command == "analyze":
        _cmd_analyze(args)
    elif args.command == "info":
        _cmd_info(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
