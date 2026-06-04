# Methodology & Theoretical Framework

This section details the rigorous methodology implemented in this repository to analyze the latent state geometry of representations.

## Dimensionality Reduction (DR) Pipeline

A central challenge in interpreting transformer hidden states is their extremely high dimensionality (e.g., $d \approx 768$ or higher). Our methodology for dimensionality reduction avoids common pitfalls, such as analyzing different subsets of data with different fitted models.

### Global Fitting 
We enforce a **Global PCA followed by Global UMAP** approach.

1. **Global PCA**: We fit a PCA model over *all* prompts across *all* layers for a given language model. Retaining the top $k$ components (e.g., $k=50$) provides an orthogonal basis that preserves the vast majority of variance while dampening noise.
2. **Global UMAP**: The dimensionally-reduced PCA coordinates are then fed into a global UMAP model (e.g., targeting 3 components). 

This ensures that the reduced space maintains global structural consistency, allowing layer $l_1$ and layer $l_m$ representations to be compared validly within the same embedding space.

## Trajectory Metrics

We define several key metrics computed purely in high-dimensional space prior to any nonlinear dimensionality reduction:

### 1. Trajectory Convergence Index (TCI)

To quantify semantic clustering akin to "attractor states," we track the distance between representations within the same semantic category versus the distance between representations in different semantic categories.

For layer $l$, let $D_{within}(l)$ denote the mean pairwise Euclidean distance of representations within the same category.
Let $D_{between}(l)$ denote the mean pairwise distance between differing categories.

The Convergence Score is formulated as:
$$Convergence(l) = D_{between}(l) - D_{within}(l)$$

A rising Convergence Index across layers mathematically proves that representations of the same conceptual category are grouping together dynamically.

### 2. Trajectory Length and Curvature

Treating the series of layer-wise representations as a geometric path, we measure the **Trajectory Length** as the sum of $L_2$ distances between adjacent layers:

$$ Length = \sum_{l=1}^{L-1} \| \mathbf{h}_{l+1} - \mathbf{h}_l \|_2 $$

Where $\mathbf{h}_l$ is the hidden state (or mean-pooled representation) at layer $l$. 
Similarly, **Trajectory Curvature** measures the deviation from a straight-line transition, helping distinguish smooth transitions from abrupt recalibrations.

## Strict Controls & Ablations

All findings are validated against controls:
- **Random Label Permutations**: To establish null distributions and show structural findings are non-random.
- **Random Embeddings**: To guarantee that metric signals are not artifacts of the architecture dimensions.

## Reproducibility & Statistical Rigor

We employ rigorous statistical techniques:
- **Bootstrapping**: We use 95% Bootstrap Confidence Intervals extensively for plotting (e.g., trajectory length or convergence over layers).
- **Non-parametric Tests**: Mann-Whitney U testing paired with Holm-Bonferroni correction ensures pairwise significance testing avoids false discovery in multiple comparisons.