# Experiment Matrix

This document outlines the systematic set of experiments to be conducted to answer the research question. By explicitly linking Models, Prompt Families, Metrics, and Controls, this matrix prevents experiment creep and ensures rigorous evaluation.

## Core Experiments

| Experiment ID | Model | Prompt Family | Metric | Control Applied | Primary Question Answered |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | GPT-2 Small | Semantic Categories | Semantic Convergence ($D(t)$) | Shuffled Layers | Do semantic concepts converge over layers? |
| **EXP-02** | GPT-2 Small | Analogies | Trajectory Curvature ($\kappa$) | Random Embeddings | Is reasoning geometry distinctly non-linear? |
| **EXP-03** | TinyLlama | Semantic Categories | Semantic Convergence ($D(t)$) | Shuffled Layers | Do GPT-2 convergence findings generalize to modern architectures? |
| **EXP-04** | TinyLlama | Analogies | Trajectory Curvature ($\kappa$) | Random Embeddings | Do GPT-2 reasoning curvature findings generalize? |
| **EXP-05** | Qwen 2.5 | Ambiguous Concepts | Trajectory Length & Convergence | Random Labels | How do stronger models handle disambiguation across layers? |
| **EXP-06** | All Models | Lexical Variations | Trajectory Stability | Shuffled Layers | Are trajectories robust to minor syntactic perturbations? |
| **EXP-07** | All Models | Reasoning Prompts | RSA | Multiple Dim. Reductions | Are reasoning geometries similar across different models? |

## Control Validations

To ensure H0 can be rigorously tested and rejected, the following dedicated control experiments will be run across the baseline model (GPT-2 Small) before expanding to others:

| Control ID | Baseline Exp | Control Condition | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **CTRL-A** | EXP-01 | Random Labels (Assigning random words to categories) | Semantic Convergence metric ($D(t)$) should show no meaningful change across layers. |
| **CTRL-B** | EXP-02 | Random Embeddings (Untrained weights) | Trajectory Curvature ($\kappa$) should reflect a random walk with no structural difference between reasoning and non-reasoning prompts. |
| **CTRL-C** | EXP-01/03 | Shuffled Layers (Computing convergence on randomly ordered layers) | Trajectory structure and convergence trends should completely disappear. |
| **CTRL-D** | EXP-07 | Multiple Dimensionality Reductions (PCA, UMAP, t-SNE) | Visualized manifolds and RSA scores must remain consistent regardless of the reduction technique used. |