"""
Trajectory Geometry of Transformer Representations – Animated Visualization
============================================================================
Creates a ~18-second, 30 fps, 1920×1080 MP4 (+ GIF) showing GPT-2 hidden
representations evolving as 3D trajectories through PCA-projected latent space.

Phases:
  1. Title card with fade-in glow          (frames   0 – 89 )
  2. Layer-by-layer trajectory build-up     (frames  90 – 299)
  3. Full view with annotations + rotation  (frames 300 – 449)
  4. Closing stats card with fade-out       (frames 450 – 539)

Usage:
    python scripts/create_animation.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless rendering
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D     # noqa: F401
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as patheffects
import warnings
warnings.filterwarnings("ignore")

# ────────────────────── configuration ──────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "results", "projected", "gpt2_pca.parquet")
OUT_DIR      = os.path.join(PROJECT_ROOT, "figures")
OUT_MP4      = os.path.join(OUT_DIR, "trajectory_animation.mp4")
OUT_GIF      = os.path.join(OUT_DIR, "trajectory_animation.gif")

FPS          = 30
DPI          = 100
FIG_W, FIG_H = 19.20, 10.80   # inches → 1920×1080 at 100 dpi (scaled by DPI)
TOTAL_FRAMES = 540             # 18 s

BG_COLOR     = "#0a0a1a"
GRID_ALPHA   = 0.08
N_STARS      = 350             # background star-field

# Neon palette per group
GROUP_COLORS = {
    "animals":   "#00ff88",
    "vehicles":  "#ff6b35",
    "emotions":  "#ff3366",
    "reasoning": "#00ccff",
    "analogies": "#aa55ff",
}
GROUP_ORDER = ["animals", "vehicles", "emotions", "reasoning", "analogies"]

# Phase boundaries
P1_END = 90    # title card
P2_END = 300   # trajectory build
P3_END = 450   # full view + annotations
P4_END = 540   # closing card

# ────────────────────── font helpers ──────────────────────
def _setup_fonts():
    """Try Inter; fall back gracefully."""
    try:
        from matplotlib.font_manager import FontProperties, findSystemFonts
        fonts = findSystemFonts()
        inter = [f for f in fonts if "inter" in os.path.basename(f).lower()]
        if inter:
            matplotlib.rcParams["font.family"] = "Inter"
            return
    except Exception:
        pass
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Segoe UI", "Helvetica Neue",
                                               "Arial", "sans-serif"]

_setup_fonts()

# ────────────────────── data loading ──────────────────────
print(f"Loading data from {DATA_PATH} ...")
df = pd.read_parquet(DATA_PATH)
print(f"  Shape : {df.shape}")
print(f"  Groups: {sorted(df['group'].unique())}")
print(f"  Layers: {sorted(df['layer'].unique())}")

# Organise trajectories: dict[group] → list of arrays (N_prompts × 12 × 3)
layers_sorted = sorted(df["layer"].unique())
N_LAYERS = len(layers_sorted)

trajectories = {}   # group → np.ndarray (n_prompts, n_layers, 3)
for grp in GROUP_ORDER:
    sub = df[df["group"] == grp].copy()
    if sub.empty:
        continue
    pids = sorted(sub["prompt_id"].unique())
    arrs = []
    for pid in pids:
        p = sub[sub["prompt_id"] == pid].sort_values("layer")
        arrs.append(p[["x", "y", "z"]].values)
    trajectories[grp] = np.array(arrs)   # (n, L, 3)

# Compute global axis limits with padding
all_xyz = df[["x", "y", "z"]].values
pad = 0.15
ranges = []
for i in range(3):
    lo, hi = all_xyz[:, i].min(), all_xyz[:, i].max()
    span = hi - lo
    ranges.append((lo - pad * span, hi + pad * span))

print(f"  Trajectories per group: { {g: t.shape[0] for g, t in trajectories.items()} }")

# ────────────────── star-field background ──────────────────
rng = np.random.default_rng(42)
star_x = rng.uniform(ranges[0][0], ranges[0][1], N_STARS)
star_y = rng.uniform(ranges[1][0], ranges[1][1], N_STARS)
star_z = rng.uniform(ranges[2][0], ranges[2][1], N_STARS)
star_sizes = rng.exponential(0.4, N_STARS) * 2

# ────────────────── layer-depth colormap ──────────────────
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def _make_line_colors(base_hex, n):
    """Generate n colours ramping brightness for layer depth."""
    r, g, b = _hex_to_rgb(base_hex)
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        # Start dimmer, end fully bright
        brightness = 0.35 + 0.65 * t
        alpha = 0.5 + 0.5 * t
        colors.append((r * brightness, g * brightness, b * brightness, alpha))
    return colors

# ────────────────── figure & axes setup ──────────────────
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG_COLOR, dpi=100)

# 3D axis – hidden initially, shown in phase 2+
ax3d = fig.add_axes([0.05, 0.02, 0.90, 0.96], projection="3d",
                     facecolor=BG_COLOR)

# Overlay axes for 2-D text (title card, annotations, closing)
ax_overlay = fig.add_axes([0, 0, 1, 1], facecolor="none")
ax_overlay.set_xlim(0, 1)
ax_overlay.set_ylim(0, 1)
ax_overlay.axis("off")

# Style 3D axis
for axis_obj in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
    axis_obj.pane.fill = False
    axis_obj.pane.set_edgecolor((1, 1, 1, GRID_ALPHA))
    axis_obj.line.set_color((1, 1, 1, 0.15))
    axis_obj._axinfo["grid"]["color"] = (1, 1, 1, GRID_ALPHA)
    axis_obj._axinfo["grid"]["linestyle"] = "-"
    axis_obj.set_tick_params(labelcolor=(1, 1, 1, 0.3), labelsize=7)

ax3d.set_xlim(*ranges[0])
ax3d.set_ylim(*ranges[1])
ax3d.set_zlim(*ranges[2])
ax3d.set_xlabel("PC 1", color="white", fontsize=9, labelpad=8, alpha=0.5)
ax3d.set_ylabel("PC 2", color="white", fontsize=9, labelpad=8, alpha=0.5)
ax3d.set_zlabel("PC 3", color="white", fontsize=9, labelpad=8, alpha=0.5)

# Glow text effect
glow_thin = [patheffects.withStroke(linewidth=3, foreground=BG_COLOR),
             patheffects.Normal()]
glow_thick = [patheffects.withStroke(linewidth=6, foreground="#1a1a3a"),
              patheffects.Normal()]
glow_neon = lambda c: [patheffects.withStroke(linewidth=8, foreground=c + "44"),
                       patheffects.withStroke(linewidth=4, foreground=c + "88"),
                       patheffects.Normal()]

# ────────────────── pre-create persistent artists ──────────────────
# Star field
star_scatter = ax3d.scatter(star_x, star_y, star_z,
                            s=star_sizes, c="white", alpha=0.0,
                            depthshade=True, edgecolors="none")

# Trajectory lines & points – we'll store handles per group
line_artists = {}   # group → list of Line3D per prompt
point_artists = {}  # group → list of scatter3D per prompt

for grp, data in trajectories.items():
    color = GROUP_COLORS.get(grp, "#ffffff")
    line_artists[grp] = []
    point_artists[grp] = []
    for p_idx in range(data.shape[0]):
        line, = ax3d.plot([], [], [], color=color, linewidth=1.8,
                          alpha=0.0, zorder=5)
        line_artists[grp].append(line)
        sc = ax3d.scatter([], [], [], s=0, c=color, edgecolors="white",
                          linewidths=0.4, alpha=0.0, depthshade=True, zorder=6)
        point_artists[grp].append(sc)

# Legend (shown in phase 2+)
legend_texts = []
for i, grp in enumerate(GROUP_ORDER):
    if grp not in trajectories:
        continue
    color = GROUP_COLORS[grp]
    txt = ax_overlay.text(0.88, 0.88 - i * 0.045, f"● {grp.capitalize()}",
                          fontsize=12, fontweight="bold",
                          color=color, alpha=0.0, ha="left", va="center",
                          path_effects=glow_neon(color),
                          transform=ax_overlay.transAxes)
    legend_texts.append(txt)

# Title card text elements
title_main = ax_overlay.text(
    0.50, 0.58,
    "Trajectory Geometry of\nTransformer Representations",
    fontsize=38, fontweight="bold", color="white", alpha=0.0,
    ha="center", va="center", linespacing=1.3,
    path_effects=glow_thick,
    transform=ax_overlay.transAxes)

title_sub = ax_overlay.text(
    0.50, 0.40,
    "How do transformer hidden states evolve across layers?",
    fontsize=18, color="#88aacc", alpha=0.0,
    ha="center", va="center", style="italic",
    path_effects=glow_thin,
    transform=ax_overlay.transAxes)

title_model = ax_overlay.text(
    0.50, 0.32,
    "GPT-2  ·  12 Layers  ·  768 Dimensions  →  3D PCA",
    fontsize=13, color="#556688", alpha=0.0,
    ha="center", va="center",
    transform=ax_overlay.transAxes)

# Annotation texts (phase 3)
ann_converge = ax_overlay.text(
    0.22, 0.82,
    "Semantic categories\nCONVERGE in deeper layers",
    fontsize=15, fontweight="bold", color="#00ff88", alpha=0.0,
    ha="center", va="center", linespacing=1.4,
    path_effects=glow_neon("#00ff88"),
    transform=ax_overlay.transAxes)

ann_diverge = ax_overlay.text(
    0.22, 0.18,
    "Reasoning prompts follow\nDIVERGENT geometric paths",
    fontsize=15, fontweight="bold", color="#00ccff", alpha=0.0,
    ha="center", va="center", linespacing=1.4,
    path_effects=glow_neon("#00ccff"),
    transform=ax_overlay.transAxes)

# Layer depth indicator
layer_label = ax_overlay.text(
    0.50, 0.04, "", fontsize=13, color="white", alpha=0.0,
    ha="center", va="center", fontweight="bold",
    path_effects=glow_thin,
    transform=ax_overlay.transAxes)

# Closing card texts
close_title = ax_overlay.text(
    0.50, 0.62,
    "Trajectory Geometry of Transformer Representations",
    fontsize=28, fontweight="bold", color="white", alpha=0.0,
    ha="center", va="center",
    path_effects=glow_thick,
    transform=ax_overlay.transAxes)

close_stats = ax_overlay.text(
    0.50, 0.48,
    "50 prompts  ×  12 layers  ×  768 dimensions  →  3D trajectories",
    fontsize=16, color="#88aacc", alpha=0.0,
    ha="center", va="center",
    path_effects=glow_thin,
    transform=ax_overlay.transAxes)

close_url = ax_overlay.text(
    0.50, 0.38,
    "github.com/latent-trajectories",
    fontsize=14, color="#556688", alpha=0.0,
    ha="center", va="center", style="italic",
    transform=ax_overlay.transAxes)

close_author = ax_overlay.text(
    0.50, 0.30,
    "Research by [Author Name]",
    fontsize=12, color="#445566", alpha=0.0,
    ha="center", va="center",
    transform=ax_overlay.transAxes)

# Colorbar for layer depth (will appear in phase 2+)
cbar_ax = fig.add_axes([0.08, 0.06, 0.20, 0.015], facecolor="none")
cbar_ax.set_visible(False)
layer_cmap = LinearSegmentedColormap.from_list("layer_depth",
    ["#1a1a4a", "#4444aa", "#00ccff", "#00ff88", "#ffffff"], N=256)
gradient = np.linspace(0, 1, 256).reshape(1, -1)
cbar_ax.imshow(gradient, aspect="auto", cmap=layer_cmap, extent=[1, 12, 0, 1])
cbar_ax.set_yticks([])
cbar_ax.set_xticks([1, 4, 8, 12])
cbar_ax.set_xticklabels(["1", "4", "8", "12"], fontsize=8, color="white")
cbar_ax.set_xlabel("Layer Depth: 1 → 12", fontsize=9, color="white", labelpad=4)
cbar_ax.tick_params(colors="white", length=3)
for spine in cbar_ax.spines.values():
    spine.set_edgecolor((1, 1, 1, 0.3))

# ────────────────── easing helpers ──────────────────
def ease_in_out(t):
    """Smooth-step easing 0→1."""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

def ease_out_cubic(t):
    t = np.clip(t, 0, 1)
    return 1 - (1 - t) ** 3

def fade_alpha(frame, start, end, fade_in=15, fade_out=15):
    """Calculate alpha with fade-in/out for a phase."""
    if frame < start or frame >= end:
        return 0.0
    elapsed = frame - start
    remaining = end - 1 - frame
    if elapsed < fade_in:
        return ease_in_out(elapsed / fade_in)
    if remaining < fade_out:
        return ease_in_out(remaining / fade_out)
    return 1.0

# ────────────────── animation function ──────────────────
def animate(frame):
    """Update all artists for a single frame."""

    # ─── PHASE 1: Title Card ─── (0–89)
    if frame < P1_END:
        ax3d.set_visible(False)
        cbar_ax.set_visible(False)

        alpha = fade_alpha(frame, 0, P1_END, fade_in=30, fade_out=20)
        title_main.set_alpha(alpha)
        title_sub.set_alpha(alpha * 0.85)
        title_model.set_alpha(alpha * 0.6)

        # Subtle glow pulsation on main title
        pulse = 0.85 + 0.15 * np.sin(frame * 0.08)
        title_main.set_fontsize(38 * pulse)

        # Hide phase 3/4 texts
        ann_converge.set_alpha(0)
        ann_diverge.set_alpha(0)
        layer_label.set_alpha(0)
        close_title.set_alpha(0)
        close_stats.set_alpha(0)
        close_url.set_alpha(0)
        close_author.set_alpha(0)
        for lt in legend_texts:
            lt.set_alpha(0)
        return

    # ─── Show 3D axis for phases 2–4 (or until phase 4 fade) ───
    ax3d.set_visible(True)

    # Hide title card
    title_main.set_alpha(0)
    title_sub.set_alpha(0)
    title_model.set_alpha(0)

    # ─── Camera rotation ───
    if frame < P4_END:
        # elevation oscillates slightly, azimuth rotates
        progress = (frame - P1_END) / (P4_END - P1_END)
        elev = 22 + 8 * np.sin(progress * np.pi * 1.5)
        azim = 30 + progress * 200  # ~200° sweep
        ax3d.view_init(elev=elev, azim=azim)

    # ─── Star field ───
    star_alpha_base = 0.25 + 0.15 * np.sin(frame * 0.05)
    if frame < P1_END + 20:
        star_alpha_base *= ease_in_out((frame - P1_END) / 20)
    # Fade out in phase 4
    if frame >= P3_END:
        star_alpha_base *= fade_alpha(frame, P3_END, P4_END, fade_in=0, fade_out=60)
    star_scatter.set_alpha(star_alpha_base)

    # ─── Colorbar visibility ───
    cbar_visible = P1_END <= frame < P3_END + 30
    cbar_ax.set_visible(cbar_visible)

    # ─── PHASE 2: Layer-by-layer build-up ─── (90–299)
    if frame < P2_END:
        # Which layer are we on?
        build_frames = P2_END - P1_END  # 210 frames for 12 layers
        frames_per_layer = build_frames / N_LAYERS
        progress = (frame - P1_END) / build_frames
        current_layer_float = progress * N_LAYERS
        current_layer = min(int(current_layer_float), N_LAYERS - 1)
        sub_progress = current_layer_float - current_layer  # 0→1 within layer

        # Update legend fade-in
        legend_alpha = ease_in_out(min(1, (frame - P1_END) / 40))
        for lt in legend_texts:
            lt.set_alpha(legend_alpha)

        # Layer label
        layer_label.set_text(f"Layer {current_layer + 1} / {N_LAYERS}")
        layer_label.set_alpha(0.8)

        # Update trajectories
        for grp, data in trajectories.items():
            color = GROUP_COLORS.get(grp, "#ffffff")
            layer_colors = _make_line_colors(color, N_LAYERS)

            for p_idx in range(data.shape[0]):
                # Points up to current layer
                n_show = current_layer + 1
                xs = data[p_idx, :n_show, 0]
                ys = data[p_idx, :n_show, 1]
                zs = data[p_idx, :n_show, 2]

                # Interpolate next point
                if current_layer < N_LAYERS - 1 and sub_progress > 0:
                    next_pt = data[p_idx, current_layer + 1]
                    curr_pt = data[p_idx, current_layer]
                    interp = curr_pt + (next_pt - curr_pt) * ease_out_cubic(sub_progress)
                    xs = np.append(xs, interp[0])
                    ys = np.append(ys, interp[1])
                    zs = np.append(zs, interp[2])

                # Line
                line = line_artists[grp][p_idx]
                line.set_data_3d(xs, ys, zs)
                line_alpha = 0.55 + 0.25 * (current_layer / max(N_LAYERS - 1, 1))
                line.set_alpha(line_alpha)
                line.set_linewidth(1.2 + 0.8 * (current_layer / max(N_LAYERS - 1, 1)))
                line.set_color(layer_colors[min(current_layer, len(layer_colors) - 1)])

                # Points – pulse newest point
                sizes = np.full(len(xs), 18.0)
                if len(sizes) > 0:
                    # Newest point pulses
                    pulse_size = 35 + 20 * np.sin(sub_progress * np.pi)
                    sizes[-1] = pulse_size
                    # Older points scale by depth
                    for k in range(len(sizes) - 1):
                        sizes[k] = 10 + 12 * (k / max(N_LAYERS - 1, 1))

                sc = point_artists[grp][p_idx]
                sc._offsets3d = (xs, ys, zs)
                sc.set_sizes(sizes)
                sc.set_alpha(line_alpha)

        # Hide phase 3/4 texts
        ann_converge.set_alpha(0)
        ann_diverge.set_alpha(0)
        close_title.set_alpha(0)
        close_stats.set_alpha(0)
        close_url.set_alpha(0)
        close_author.set_alpha(0)
        return

    # ─── PHASE 3: Full view + Annotations ─── (300–449)
    if frame < P3_END:
        # All trajectories fully visible
        for grp, data in trajectories.items():
            color = GROUP_COLORS.get(grp, "#ffffff")
            layer_colors = _make_line_colors(color, N_LAYERS)

            for p_idx in range(data.shape[0]):
                xs = data[p_idx, :, 0]
                ys = data[p_idx, :, 1]
                zs = data[p_idx, :, 2]

                line = line_artists[grp][p_idx]
                line.set_data_3d(xs, ys, zs)
                line.set_alpha(0.75)
                line.set_linewidth(2.0)
                line.set_color(layer_colors[-1])

                sizes = np.array([10 + 18 * (k / max(N_LAYERS - 1, 1))
                                  for k in range(N_LAYERS)])
                sc = point_artists[grp][p_idx]
                sc._offsets3d = (xs, ys, zs)
                sc.set_sizes(sizes)
                sc.set_alpha(0.85)

        # Legend
        for lt in legend_texts:
            lt.set_alpha(1.0)

        # Layer label shows range
        layer_label.set_text("Layers 1 → 12")
        layer_label.set_alpha(0.7)

        # Annotations fade in
        ann_progress = (frame - P2_END) / 60  # fade over 2 sec
        ann_alpha = ease_in_out(min(1, ann_progress))

        # Fade out annotations near end of phase 3
        if frame > P3_END - 40:
            ann_alpha *= fade_alpha(frame, P3_END - 40, P3_END, fade_in=0, fade_out=30)

        ann_converge.set_alpha(ann_alpha)
        ann_diverge.set_alpha(ann_alpha)

        # Hide phase 4
        close_title.set_alpha(0)
        close_stats.set_alpha(0)
        close_url.set_alpha(0)
        close_author.set_alpha(0)
        return

    # ─── PHASE 4: Closing Card ─── (450–539)
    # Fade out 3D scene
    scene_alpha = fade_alpha(frame, P3_END, P3_END + 40, fade_in=0, fade_out=40)
    for grp in trajectories:
        for p_idx in range(len(line_artists[grp])):
            line_artists[grp][p_idx].set_alpha(scene_alpha * 0.75)
            point_artists[grp][p_idx].set_alpha(scene_alpha * 0.85)
    for lt in legend_texts:
        lt.set_alpha(scene_alpha)
    layer_label.set_alpha(0)
    ann_converge.set_alpha(0)
    ann_diverge.set_alpha(0)
    cbar_ax.set_visible(False)

    # Fade in closing text
    close_alpha = fade_alpha(frame, P3_END + 20, P4_END, fade_in=30, fade_out=25)
    close_title.set_alpha(close_alpha)
    close_stats.set_alpha(close_alpha * 0.85)
    close_url.set_alpha(close_alpha * 0.65)
    close_author.set_alpha(close_alpha * 0.5)

# ────────────────── render ──────────────────
print(f"\nRendering {TOTAL_FRAMES} frames @ {FPS} fps = {TOTAL_FRAMES/FPS:.1f}s ...")
print(f"  Output MP4: {OUT_MP4}")
print(f"  Output GIF: {OUT_GIF}")
os.makedirs(OUT_DIR, exist_ok=True)

anim = animation.FuncAnimation(fig, animate, frames=TOTAL_FRAMES,
                                interval=1000 // FPS, blit=False)

# Try ffmpeg first for MP4
mp4_saved = False
try:
    writer_mp4 = animation.FFMpegWriter(fps=FPS, bitrate=5000,
                                         codec="libx264",
                                         extra_args=["-pix_fmt", "yuv420p"])
    anim.save(OUT_MP4, writer=writer_mp4, dpi=DPI)
    print(f"  [OK] MP4 saved: {OUT_MP4}")
    mp4_saved = True
except Exception as e:
    print(f"  [WARN] FFmpeg not available ({type(e).__name__}), skipping MP4.")

# Save GIF via Pillow (lower DPI to keep file size manageable)
print("  Saving GIF (this may take a few minutes)...")
try:
    writer_gif = animation.PillowWriter(fps=min(FPS, 15))  # lower fps for GIF
    gif_dpi = int(DPI * 0.5)  # 75 DPI for GIF
    anim.save(OUT_GIF, writer=writer_gif, dpi=gif_dpi)
    print(f"  [OK] GIF saved: {OUT_GIF}")
except Exception as e:
    print(f"  [WARN] GIF save failed: {type(e).__name__}: {e}")

if not mp4_saved:
    # Also save a full-res GIF as the "primary" output
    try:
        fallback_gif = OUT_MP4.replace(".mp4", ".gif")
        if fallback_gif != OUT_GIF:
            writer_gif2 = animation.PillowWriter(fps=FPS)
            anim.save(fallback_gif, writer=writer_gif2, dpi=int(DPI * 0.75))
            print(f"  [OK] Full-res GIF saved as: {fallback_gif}")
    except Exception as e:
        print(f"  [WARN] Full-res GIF also failed: {type(e).__name__}: {e}")

print("\nDone! Animation rendering complete.")
plt.close("all")
