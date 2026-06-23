"""
Latent Trajectories — HuggingFace Spaces Gradio Demo.

Interactive geometric analysis of transformer hidden-state trajectories.
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import json
import time

from latent_trajectories import GeometryProbe, ProbeResult

# ── Globals ──────────────────────────────────────────────────────────────

_CACHED_PROBES = {}

# ── Color palette (Catppuccin Mocha) ────────────────────────────────────

COLORS = [
    "#89b4fa",  # blue
    "#f38ba8",  # red
    "#a6e3a1",  # green
    "#fab387",  # peach
    "#cba6f7",  # mauve
    "#f9e2af",  # yellow
    "#94e2d5",  # teal
    "#f5c2e7",  # pink
    "#74c7ec",  # sapphire
    "#eba0ac",  # maroon
]

BG_COLOR = "#1e1e2e"
GRID_COLOR = "#45475a"
TEXT_COLOR = "#cdd6f4"
SURFACE_COLOR = "#313244"


def _get_probe(model_name: str) -> GeometryProbe:
    """Cache probes to avoid reloading models."""
    if model_name not in _CACHED_PROBES:
        _CACHED_PROBES[model_name] = GeometryProbe(model_name, device="cpu")
    return _CACHED_PROBES[model_name]


def _build_3d_plot(result: ProbeResult) -> go.Figure:
    """Create an interactive 3D Plotly trajectory plot."""
    trajectories = result.trajectories
    labels = result.labels or [f"text_{i}" for i in range(len(trajectories))]

    # Flatten all hidden states for PCA fitting
    all_states = []
    for t in trajectories:
        tensor = t.hidden_states
        if hasattr(tensor, "numpy"):
            tensor = tensor.detach().cpu().numpy()
        # Shape: [num_layers, hidden_dim] — take mean over seq_len if needed
        if tensor.ndim == 3:
            tensor = tensor.mean(axis=1)
        all_states.append(tensor)

    all_flat = np.vstack(all_states)
    n_components = min(3, all_flat.shape[1], all_flat.shape[0])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(all_flat)

    # Split back into individual trajectories
    fig = go.Figure()
    unique_labels = list(dict.fromkeys(labels))
    label_colors = {lbl: COLORS[i % len(COLORS)] for i, lbl in enumerate(unique_labels)}

    idx = 0
    for i, t in enumerate(trajectories):
        tensor = t.hidden_states
        if hasattr(tensor, "numpy"):
            tensor = tensor.detach().cpu().numpy()
        if tensor.ndim == 3:
            tensor = tensor.mean(axis=1)
        n_layers = tensor.shape[0]
        pts = projected[idx : idx + n_layers]
        idx += n_layers

        label = labels[i]
        color = label_colors[label]

        # Trajectory line
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0] if n_components >= 1 else np.zeros(n_layers),
                y=pts[:, 1] if n_components >= 2 else np.zeros(n_layers),
                z=pts[:, 2] if n_components >= 3 else np.zeros(n_layers),
                mode="lines+markers",
                name=f"{label}",
                legendgroup=label,
                showlegend=(i == labels.index(label)),
                line=dict(color=color, width=4),
                marker=dict(
                    size=4,
                    color=list(range(n_layers)),
                    colorscale=[
                        [0, color],
                        [1, "#ffffff"],
                    ],
                    opacity=0.8,
                ),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Layer %{customdata}<br>"
                    "PC1: %{x:.2f}<br>"
                    "PC2: %{y:.2f}<br>"
                    "PC3: %{z:.2f}<extra></extra>"
                ),
                customdata=list(range(n_layers)),
            )
        )

        # Start marker
        fig.add_trace(
            go.Scatter3d(
                x=[pts[0, 0]] if n_components >= 1 else [0],
                y=[pts[0, 1]] if n_components >= 2 else [0],
                z=[pts[0, 2]] if n_components >= 3 else [0],
                mode="markers",
                marker=dict(size=8, color=color, symbol="diamond", opacity=1.0),
                legendgroup=label,
                showlegend=False,
                hovertemplate=f"<b>{label}</b> — Layer 0 (start)<extra></extra>",
            )
        )

    # Explained variance annotation
    var_text = " + ".join(
        [f"PC{i+1}: {v:.0%}" for i, v in enumerate(pca.explained_variance_ratio_)]
    )

    fig.update_layout(
        title=dict(
            text=f"Hidden-State Trajectories ({result.model_name})",
            font=dict(size=18, color=TEXT_COLOR),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title="PC1",
                backgroundcolor=BG_COLOR,
                gridcolor=GRID_COLOR,
                color=TEXT_COLOR,
            ),
            yaxis=dict(
                title="PC2",
                backgroundcolor=BG_COLOR,
                gridcolor=GRID_COLOR,
                color=TEXT_COLOR,
            ),
            zaxis=dict(
                title="PC3",
                backgroundcolor=BG_COLOR,
                gridcolor=GRID_COLOR,
                color=TEXT_COLOR,
            ),
            bgcolor=BG_COLOR,
        ),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        legend=dict(
            bgcolor=SURFACE_COLOR,
            bordercolor=GRID_COLOR,
            font=dict(color=TEXT_COLOR),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=550,
        annotations=[
            dict(
                text=f"Explained variance: {var_text}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.02,
                showarrow=False,
                font=dict(size=11, color="#a6adc8"),
            )
        ],
    )

    return fig


def _format_metrics(result: ProbeResult) -> str:
    """Format metrics into a readable markdown string."""
    m = result.metrics
    summary = m.get("summary", {})

    lines = [
        "## 📊 Metrics Summary\n",
        f"| Property | Value |",
        f"|---|---|",
        f"| **Model** | `{result.model_name}` |",
        f"| **Trajectories** | {summary.get('n_trajectories', 'N/A')} |",
        f"| **Layers** | {summary.get('n_layers', 'N/A')} |",
        f"| **Hidden dim** | {summary.get('hidden_dim', 'N/A')} |",
        f"| **Mean trajectory length** | {summary.get('mean_trajectory_length', 'N/A'):.4f} |",
        f"| **Mean curvature** | {summary.get('mean_curvature', 'N/A'):.4f} |",
        "",
        "### Per-Trajectory Lengths",
        "| Index | Label | Length | Curvature |",
        "|---|---|---|---|",
    ]

    lengths = m.get("trajectory_length", [])
    curvatures = m.get("curvature", [])
    labels = result.labels or [f"text_{i}" for i in range(len(lengths))]

    for i, (l, c, lbl) in enumerate(zip(lengths, curvatures, labels)):
        lines.append(f"| {i} | {lbl} | {l:.4f} | {c:.4f} |")

    return "\n".join(lines)


def _format_controls(controls: dict) -> str:
    """Format controls results into markdown."""
    lines = ["## 🧪 Statistical Controls\n"]

    # Label permutation
    lp = controls.get("label_permutation", {})
    passed = lp.get("passed")
    if passed is None:
        lines.append("### 1. Label Permutation — ⏭️ Skipped (no labels provided)")
    else:
        icon = "✅" if passed else "❌"
        lines.append(f"### 1. Label Permutation — {icon} {'Passed' if passed else 'Failed'}")
        lines.append(f"- Significant layers: {lp.get('layers_significant', 'N/A')} / {lp.get('total_layers', 'N/A')}")

    # Gaussian noise
    gn = controls.get("gaussian_noise", {})
    gn_passed = gn.get("passed")
    if gn_passed is not None:
        icon = "✅" if gn_passed else "❌"
        lines.append(f"\n### 2. Gaussian Noise — {icon} {'Passed' if gn_passed else 'Failed'}")
        lines.append(f"- Real trajectory length (mean): {gn.get('real_length_mean', 'N/A'):.4f}")
        lines.append(f"- Null trajectory length (mean): {gn.get('null_length_mean', 'N/A'):.4f}")

    # Temporal shuffle
    ts = controls.get("temporal_shuffle", {})
    ts_passed = ts.get("passed")
    if ts_passed is not None:
        icon = "✅" if ts_passed else "❌"
        lines.append(f"\n### 3. Temporal Shuffle — {icon} {'Passed' if ts_passed else 'Failed'}")
        lines.append(f"- Real curvature (mean): {ts.get('real_curvature_mean', 'N/A'):.4f}")
        lines.append(f"- Null curvature (mean): {ts.get('null_curvature_mean', 'N/A'):.4f}")

    return "\n".join(lines)


def analyze(
    model_name: str,
    texts_input: str,
    labels_input: str,
    run_controls: bool,
    progress=gr.Progress(),
):
    """Main analysis function called by Gradio."""
    # Parse inputs
    texts = [t.strip() for t in texts_input.strip().split("\n") if t.strip()]
    if not texts:
        raise gr.Error("Please enter at least one text prompt.")

    labels = None
    if labels_input.strip():
        labels = [l.strip() for l in labels_input.strip().split("\n") if l.strip()]
        if len(labels) != len(texts):
            raise gr.Error(
                f"Number of labels ({len(labels)}) must match "
                f"number of texts ({len(texts)}). Got {len(texts)} texts."
            )

    # Load model
    progress(0.1, desc="Loading model...")
    probe = _get_probe(model_name)

    # Run analysis
    progress(0.3, desc="Extracting hidden states...")
    result = probe.run(texts, labels=labels)

    # Build outputs
    progress(0.7, desc="Building visualization...")
    plot = _build_3d_plot(result)
    metrics_md = _format_metrics(result)

    # Controls (optional)
    controls_md = ""
    if run_controls:
        progress(0.85, desc="Running statistical controls...")
        controls = result.controls(num_permutations=500, seed=42)
        controls_md = _format_controls(controls)

    progress(1.0, desc="Done!")
    return plot, metrics_md, controls_md


# ── Example prompts ─────────────────────────────────────────────────────

EXAMPLE_TEXTS = """The capital of France is Paris.
What is the square root of 144?
Write a haiku about autumn leaves.
The speed of light is approximately 300,000 km/s.
Explain why the sky appears blue.
Once upon a time in a land far away..."""

EXAMPLE_LABELS = """factual
reasoning
creative
factual
reasoning
creative"""


# ── Gradio UI ────────────────────────────────────────────────────────────

DESCRIPTION = """
# 🧬 Latent Trajectories

**Geometric analysis of transformer hidden-state trajectories across layers.**

When a transformer processes text, each layer transforms the hidden representation — 
tracing a *trajectory* through high-dimensional space. This tool measures the **geometry** 
of that trajectory: how far it travels, how sharply it bends, and whether semantically 
related inputs converge.

> 💡 **How to use**: Enter one text per line below, optionally with matching labels. 
> Click **Analyze** to see 3D trajectory visualizations and geometric metrics.

📦 `pip install latent-trajectories` · 
[GitHub](https://github.com/Vishal-sys-code/latent-trajectories) · 
[Paper methodology](https://github.com/Vishal-sys-code/latent-trajectories/blob/main/docs/research_spec.md)
"""

CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
footer { display: none !important; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
    title="Latent Trajectories — Geometric Analysis of Transformer Hidden States",
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            model_name = gr.Dropdown(
                choices=["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"],
                value="gpt2",
                label="🤖 Model",
                info="Select a HuggingFace causal LM. GPT-2 loads fastest on CPU.",
            )
            texts_input = gr.Textbox(
                label="📝 Texts (one per line)",
                placeholder="Enter prompts, one per line...",
                lines=8,
                value=EXAMPLE_TEXTS,
            )
            labels_input = gr.Textbox(
                label="🏷️ Labels (optional, one per line)",
                placeholder="Enter labels matching each text line...",
                lines=8,
                value=EXAMPLE_LABELS,
            )
            run_controls = gr.Checkbox(
                label="🧪 Run statistical controls",
                value=True,
                info="Runs label permutation, Gaussian noise, and temporal shuffle controls. Takes ~30s extra.",
            )
            analyze_btn = gr.Button(
                "🔬 Analyze Trajectories",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=2):
            plot_output = gr.Plot(
                label="3D Trajectory Visualization",
            )
            with gr.Row():
                metrics_output = gr.Markdown(
                    label="Metrics",
                    value="*Click Analyze to see metrics.*",
                )
                controls_output = gr.Markdown(
                    label="Controls",
                    value="*Enable controls checkbox and click Analyze.*",
                )

    analyze_btn.click(
        fn=analyze,
        inputs=[model_name, texts_input, labels_input, run_controls],
        outputs=[plot_output, metrics_output, controls_output],
    )

    gr.Markdown(
        """
---
**How it works**: Each text is fed through the transformer. The hidden states at every layer 
form a trajectory in $d$-dimensional space. We measure trajectory **length** (total distance), 
**curvature** (bending), and **convergence** (clustering by label). PCA projects to 3D for visualization.

Built by [Vishal](https://github.com/Vishal-sys-code) · MIT License · 
[Research Spec](https://github.com/Vishal-sys-code/latent-trajectories/blob/main/docs/research_spec.md)
"""
    )


if __name__ == "__main__":
    demo.launch()
