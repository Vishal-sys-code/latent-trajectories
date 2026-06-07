"""
Figure 6 Replacement: 2D PCA Overlay of Trajectory Keyframes
=============================================================
Generates publication-quality 2D scatter plots showing all five
selected layers overlaid in the same PCA(2) coordinate system
with colored trajectory lines connecting each prompt's
representations across layers.

Generates figures for ALL three models:
  - GPT-2       (12 layers, d=768)
  - TinyLlama   (22 layers, d=2048)
  - Qwen2.5     (28 layers, d=1536)

If a model lacks a pre-projected parquet file, the script projects
directly from raw .pt trajectory files using PCA(2).

Output:
    figures/figure6_keyframe_pca_overlay_gpt2.png
    figures/figure6_keyframe_pca_overlay_tinyllama.png
    figures/figure6_keyframe_pca_overlay_qwen2.5.png
    paper/figure6_keyframe_pca_overlay.png   (GPT-2 copy for LaTeX)

Usage:
    python scripts/plot_keyframe_pca_overlay.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# ────────────────────── paths ──────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECTED_DIR = os.path.join(PROJECT_ROOT, "results", "projected")
TRAJ_DIR      = os.path.join(PROJECT_ROOT, "data", "trajectories")
PROMPTS_FILE  = os.path.join(PROJECT_ROOT, "data", "prompts", "prompts.jsonl")
OUT_DIR       = os.path.join(PROJECT_ROOT, "figures")
PAPER_DIR     = os.path.join(PROJECT_ROOT, "paper")

# ────────────────────── model configs ──────────────────────
MODELS = {
    "gpt2": {
        "display": "GPT-2 Small",
        "layers": 12,
        "traj_dir": os.path.join(TRAJ_DIR, "gpt2"),
        "parquet": os.path.join(PROJECTED_DIR, "gpt2_pca.parquet"),
    },
    "tinyllama": {
        "display": "TinyLlama-1.1B",
        "layers": 22,
        "traj_dir": os.path.join(TRAJ_DIR, "tinyllama"),
        "parquet": os.path.join(PROJECTED_DIR, "tinyllama_pca.parquet"),
    },
    "qwen2.5": {
        "display": "Qwen2.5-1.5B",
        "layers": 28,
        "traj_dir": os.path.join(TRAJ_DIR, "qwen2.5"),
        "parquet": os.path.join(PROJECTED_DIR, "qwen2.5_pca.parquet"),
    },
}

# ────────────────────── visual config ──────────────────────
GROUP_COLORS = {
    "animals":   "#2ecc71",
    "vehicles":  "#e67e22",
    "objects":   "#e67e22",   # same hue as vehicles (equivalent category)
    "emotions":  "#e74c3c",
    "reasoning": "#3498db",
    "analogies": "#9b59b6",
}
GROUP_ORDER = ["animals", "vehicles", "objects", "emotions", "reasoning", "analogies"]
GROUP_MARKERS = {
    "animals":   "o",
    "vehicles":  "s",
    "objects":   "s",         # same marker as vehicles
    "emotions":  "D",
    "reasoning": "^",
    "analogies": "v",
}

LAYER_CMAP = LinearSegmentedColormap.from_list(
    "layer_depth",
    ["#2c1654", "#3a6ea5", "#2ecc71", "#f1c40f", "#ff6b35"],
    N=256,
)

# ────────────────────── font setup ──────────────────────
def _setup_fonts():
    try:
        from matplotlib.font_manager import findSystemFonts
        fonts = findSystemFonts()
        inter = [f for f in fonts if "inter" in os.path.basename(f).lower()]
        if inter:
            matplotlib.rcParams["font.family"] = "Inter"
            return
    except Exception:
        pass
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"
    ]

_setup_fonts()

# ────────────────────── projection helper ──────────────────────
def project_from_trajectories(model_key: str, cfg: dict) -> pd.DataFrame:
    """Load raw .pt trajectories, PCA-project to 2D, return a DataFrame."""
    import json
    import torch
    from sklearn.decomposition import PCA

    sys.path.insert(0, PROJECT_ROOT)
    from src.trajectories import HiddenStateTrajectory

    # Load prompt metadata
    prompts_data = {}
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                prompts_data[str(p["id"])] = p

    # Load all trajectories
    traj_dir = cfg["traj_dir"]
    pt_files = sorted([
        os.path.join(traj_dir, f) for f in os.listdir(traj_dir)
        if f.endswith(".pt") and "metadata" not in f
    ])

    trajectories = []
    for fp in pt_files:
        try:
            traj = HiddenStateTrajectory.load(fp)
            trajectories.append(traj)
        except Exception as e:
            print(f"    [WARN] Failed to load {fp}: {e}")

    if not trajectories:
        print(f"    No trajectories found for {model_key}")
        return pd.DataFrame()

    print(f"    Loaded {len(trajectories)} trajectories")

    # Stack all hidden states for global PCA fit
    all_states = []
    for t in trajectories:
        all_states.append(t.trajectory.numpy())
    X = np.concatenate(all_states, axis=0)
    print(f"    Stacked states: {X.shape}")

    # Fit PCA(2)
    pca = PCA(n_components=2)
    pca.fit(X)

    # Project each trajectory
    records = []
    for t in trajectories:
        pid = str(t.prompt_id)
        p_data = prompts_data.get(pid, {})
        group = p_data.get("group", "unknown")
        subcategory = p_data.get("subcategory", "unknown")

        states = t.trajectory.numpy()
        projected = pca.transform(states)

        for layer_idx in range(states.shape[0]):
            records.append({
                "model": model_key,
                "prompt_id": t.prompt_id,
                "group": group,
                "subcategory": subcategory,
                "layer": layer_idx + 1,
                "x": projected[layer_idx, 0],
                "y": projected[layer_idx, 1],
            })

    df = pd.DataFrame(records)

    # Save for future use
    os.makedirs(PROJECTED_DIR, exist_ok=True)
    out_path = cfg["parquet"]
    df.to_parquet(out_path, index=False)
    print(f"    Saved projection: {out_path}")

    return df


def load_or_project(model_key: str, cfg: dict) -> pd.DataFrame:
    """Load existing parquet or project from raw trajectories."""
    parquet = cfg["parquet"]
    if os.path.exists(parquet):
        print(f"  Loading existing: {parquet}")
        df = pd.read_parquet(parquet)
        # Ensure we have at least x, y columns
        if "x" in df.columns and "y" in df.columns:
            return df

    print(f"  Projecting from raw trajectories for {model_key}...")
    return project_from_trajectories(model_key, cfg)


# ────────────────────── figure generation ──────────────────────
def pick_keyframe_layers(all_layers: list, total_layers: int) -> list:
    """Pick 5 evenly-spaced keyframe layers from the available set."""
    n = len(all_layers)
    if n <= 5:
        return list(all_layers)
    idxs = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    return sorted(list(dict.fromkeys([all_layers[i] for i in idxs])))


def generate_figure(df: pd.DataFrame, model_key: str, display_name: str,
                    out_path: str, dpi: int = 300):
    """Generate a single 2D PCA overlay figure for one model."""

    all_layers = sorted(int(l) for l in df["layer"].unique())
    keyframe_layers = [int(l) for l in pick_keyframe_layers(all_layers, len(all_layers))]
    print(f"  Keyframe layers: {keyframe_layers}")

    layer_norm = Normalize(vmin=min(all_layers), vmax=max(all_layers))
    df_key = df[df["layer"].isin(keyframe_layers)].copy()

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    ax.set_facecolor("#fafafa")

    # Trajectory lines + scatter per group
    for grp in GROUP_ORDER:
        sub = df_key[df_key["group"] == grp]
        if sub.empty:
            continue
        color = GROUP_COLORS[grp]
        marker = GROUP_MARKERS[grp]
        prompt_ids = sorted(sub["prompt_id"].unique())

        # Per-prompt trajectory lines
        for pid in prompt_ids:
            p = sub[sub["prompt_id"] == pid].sort_values("layer")
            xs, ys = p["x"].values, p["y"].values
            layers = p["layer"].values
            if len(xs) >= 2:
                points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                seg_layers = (layers[:-1] + layers[1:]) / 2.0
                lc = LineCollection(
                    segments,
                    colors=[LAYER_CMAP(layer_norm(l)) for l in seg_layers],
                    linewidths=1.0, alpha=0.35, zorder=1,
                )
                ax.add_collection(lc)

        # Scatter per keyframe layer
        for lv in keyframe_layers:
            lsub = sub[sub["layer"] == lv]
            if lsub.empty:
                continue
            t = layer_norm(lv)
            ax.scatter(
                lsub["x"].values, lsub["y"].values,
                c=[LAYER_CMAP(layer_norm(lv))],
                marker=marker,
                s=25 + 80 * t,
                edgecolors=color, linewidths=0.6,
                alpha=0.7 + 0.25 * t,
                zorder=3 + int(t * 10),
            )

    # ── Convex hulls (first / last keyframe layer) ──
    from scipy.spatial import ConvexHull

    for grp in GROUP_ORDER:
        sub = df_key[df_key["group"] == grp]
        if sub.empty:
            continue
        color = GROUP_COLORS[grp]

        for lv, style, alpha_line, alpha_fill, lw in [
            (keyframe_layers[0],  "--", 0.15, 0.03, 1.0),
            (keyframe_layers[-1], "-",  0.45, 0.08, 1.5),
        ]:
            pts_df = sub[sub["layer"] == lv]
            if len(pts_df) >= 3:
                pts = pts_df[["x", "y"]].values
                try:
                    hull = ConvexHull(pts)
                    hv = np.append(hull.vertices, hull.vertices[0])
                    ax.plot(pts[hv, 0], pts[hv, 1],
                            color=color, alpha=alpha_line, linewidth=lw,
                            linestyle=style, zorder=2)
                    ax.fill(pts[hv, 0], pts[hv, 1],
                            color=color, alpha=alpha_fill, zorder=0)
                except Exception:
                    pass

    # ── Convergence arrow ──
    ref_group = "animals"
    if ref_group in df_key["group"].values:
        early = df_key[(df_key["group"] == ref_group) &
                       (df_key["layer"] == keyframe_layers[0])]
        late = df_key[(df_key["group"] == ref_group) &
                      (df_key["layer"] == keyframe_layers[-1])]
        if not early.empty and not late.empty:
            ax.annotate(
                "convergence",
                xy=(late["x"].mean(), late["y"].mean()),
                xytext=(early["x"].mean(), early["y"].mean()),
                fontsize=9, color="#2c3e50", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.5,
                                connectionstyle="arc3,rad=-0.2"),
                zorder=20, ha="center",
            )

    # ── BOTH legends in top-right ──
    group_handles = []
    for grp in GROUP_ORDER:
        if grp not in df_key["group"].values:
            continue
        group_handles.append(Line2D(
            [0], [0], marker=GROUP_MARKERS[grp], color="none",
            markerfacecolor=GROUP_COLORS[grp],
            markeredgecolor=GROUP_COLORS[grp],
            markersize=8, linewidth=0,
            label=grp.capitalize(),
        ))

    layer_handles = []
    for lv in keyframe_layers:
        c = LAYER_CMAP(layer_norm(lv))
        layer_handles.append(Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=c, markeredgecolor="grey",
            markersize=6 + 4 * layer_norm(lv), linewidth=0,
            label=f"Layer {int(lv)}",
        ))

    # Legend placement: upper-right for GPT-2, lower-right for others
    if model_key == "gpt2":
        layer_loc = "upper right"
        group_anchor_y = 0.56
    else:
        layer_loc = "lower right"
        group_anchor_y = 0.28

    leg_layer = ax.legend(
        handles=layer_handles,
        loc=layer_loc,
        frameon=True, framealpha=0.92, edgecolor="#cccccc",
        fontsize=9, title="Layer Depth", title_fontsize=10,
        handletextpad=0.5, borderpad=0.6,
    )
    ax.add_artist(leg_layer)

    # Place group legend just below the layer legend, also on the right
    leg_group = ax.legend(
        handles=group_handles,
        loc="right",
        bbox_to_anchor=(1.0, group_anchor_y),
        frameon=True, framealpha=0.92, edgecolor="#cccccc",
        fontsize=10, title="Semantic Group", title_fontsize=10,
        handletextpad=0.5, borderpad=0.6,
    )
    ax.add_artist(leg_layer)  # re-add (matplotlib quirk)

    # ── Axes formatting ──
    ax.set_xlabel("PC 1", fontsize=12, labelpad=8)
    ax.set_ylabel("PC 2", fontsize=12, labelpad=8)
    ax.set_title(
        f"Trajectory Evolution Across Layers — 2D PCA Overlay ({display_name})",
        fontsize=14, fontweight="bold", pad=14,
    )
    kf_str = ", ".join(str(int(l)) for l in keyframe_layers)
    fig.text(
        0.5, 0.915,
        f"Keyframe layers [{kf_str}]  ·  "
        f"Dashed hulls = layer {int(keyframe_layers[0])} (dispersed)  ·  "
        f"Solid hulls = layer {int(keyframe_layers[-1])} (converged)",
        ha="center", va="center", fontsize=9, color="#666666", style="italic",
    )
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#555555", labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


# ────────────────────── main ──────────────────────
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)

    for model_key, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Model: {cfg['display']}  ({model_key})")
        print(f"{'='*60}")

        df = load_or_project(model_key, cfg)
        if df.empty:
            print(f"  [SKIP] No data for {model_key}")
            continue

        print(f"  Shape: {df.shape}")
        print(f"  Groups: {sorted(df['group'].unique())}")
        print(f"  Layers: {sorted(df['layer'].unique())}")

        # Generate per-model figure
        fig_name = f"figure6_keyframe_pca_overlay_{model_key}.png"
        fig_path = os.path.join(OUT_DIR, fig_name)
        generate_figure(df, model_key, cfg["display"], fig_path)

        # Copy GPT-2 figure to paper/ for LaTeX
        if model_key == "gpt2":
            paper_path = os.path.join(PAPER_DIR, "figure6_keyframe_pca_overlay.png")
            fig_path2 = fig_path  # re-save
            generate_figure(df, model_key, cfg["display"], paper_path)

    print("\nDone — all models processed.")
