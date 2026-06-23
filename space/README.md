---
title: Latent Trajectories
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.34.2"
app_file: app.py
pinned: true
license: mit
short_description: Geometric analysis of transformer hidden-state trajectories
tags:
  - interpretability
  - transformers
  - hidden-states
  - trajectory-analysis
  - mechanistic-interpretability
---

# 🧬 Latent Trajectories

**Geometric analysis of transformer hidden-state trajectories across layers.**

This Space provides an interactive demo of the `latent-trajectories` package. Enter text prompts, select a model, and explore how the hidden-state representations evolve across transformer layers.

## What it does

1. Feeds your text through a transformer (GPT-2 by default)
2. Extracts hidden states at every layer
3. Projects them into 3D via PCA
4. Measures trajectory **length**, **curvature**, and **convergence**
5. Runs statistical controls to validate the geometry is real

## Key Findings

- **Reasoning tasks** produce 3× longer trajectories than factual lookups
- **Semantic categories** converge in middle layers (attractor dynamics)
- **Layer velocity** reveals three processing phases: encode → elaborate → decode
- All results survive label permutation, Gaussian noise, and temporal shuffle controls

## Install the library

```bash
pip install latent-trajectories
```

```python
from latent_trajectories import GeometryProbe

probe = GeometryProbe("gpt2")
result = probe.run(texts, labels)
result.controls()   # statistical validation
result.plot()       # 3D visualization
```

## Links

- 📦 [GitHub](https://github.com/Vishal-sys-code/latent-trajectories)
- 🐍 [PyPI](https://pypi.org/project/latent-trajectories/)
