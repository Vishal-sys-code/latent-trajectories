=================================================================
Trajectory Geometry of Transformer Representations
=================================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   methodology
   experiments
   api

Introduction: A Computational Neuroscience Perspective
==========================================================

This documentation provides an extensive, in-depth view of the **Trajectory Geometry of Transformer Representations Across Layers**. This project applies a highly rigorous, computationally-driven analytical framework to investigate the hidden state geometry of transformer architectures. 

While the subjects of our study are artificial neural networks (LLMs like GPT-2, TinyLlama, and Qwen2.5), we draw heavy inspiration from **computational neuroscience** to understand how hidden states evolve dynamically.

Rather than viewing transformers as static input-output mappings, we treat the forward pass as a **population trajectory** moving through a high-dimensional state space. Specifically, we investigate whether hidden-state evolution exhibits structures analogous to:

- **Neural Manifolds**: Do transformer hidden states organize into low-dimensional manifolds that govern computation?
- **Attractor Dynamics**: Do representations of semantically related concepts converge toward stable attractor states deeper in the network?
- **Population Trajectories**: Can layer-by-layer transitions be mapped and analyzed structurally as non-random geometric paths?
- **Representational Geometry**: Can the distance and curvature of these trajectories reveal reasoning, disambiguation, and semantic clustering that is invisible when looking at individual activations alone?

*Note: This framing serves as an analytical lens. We explicitly avoid claims that LLMs implement biological cognition; rather, we demonstrate that tools from neuroscience reveal profound, non-random structural organization in artificial representations.*

Key Visualizations
==================

.. figure:: ../figures/trajectory_animation.gif
   :align: center
   :alt: Trajectory Evolution Animation
   :width: 600px

   **Figure 1:** Latent state trajectories evolving across the layers of a transformer network.

Pipeline Overview
=================

Our rigorous methodology transitions from extracting hidden states to projecting high-dimensional distances into comprehensive dimensionality reductions and statistical proofs.

.. figure:: ../figures/phase_9/figure1_pipeline.png
   :align: center
   :alt: Methodology Pipeline
   :width: 800px

   **Figure 2:** Our systematic pipeline: From hidden state extraction across layers, to high-dimensional metric computation, rigorous dimensionality reduction, and statistical validation.

Neural Manifolds & Semantic Convergence
=======================================

Through Representational Similarity Analysis (RSA) and rigorously controlled Dimensionality Reduction (global PCA and UMAP), we observe that prompts within the same semantic category (e.g., "Animals") start dispersed but converge into tight, distinct regions of the latent space.

.. image:: ../figures/pca/animals_overlay.png
   :width: 49 %
   :alt: PCA Semantic Convergence
.. image:: ../figures/umap/animals_overlay.png
   :width: 49 %
   :alt: UMAP Semantic Convergence

*Figure 3: Global PCA (Left) and UMAP (Right) projections showing how representations belonging to the 'Animals' category evolve across layers. The trajectories demonstrate structured flow rather than random walks.*

Quantitative Geometry
=====================

We move beyond visual plots by employing quantitative, high-dimensional metrics:
- **Convergence Index:** Measures $D_{between}(l) - D_{within}(l)$. We observe a sharp statistical increase in convergence in the middle-to-late layers.
- **Trajectory Length:** Measures the $L_2$ distance traveled across the latent space. We find that structured reasoning tasks travel significantly longer, more curved paths compared to basic semantic lookups.

.. image:: ../figures/convergence_score_layers.png
   :width: 49 %
   :alt: Convergence Index
.. image:: ../figures/figure6_trajectory_length.png
   :width: 49 %
   :alt: Trajectory Length

*Figure 4: (Left) The Trajectory Convergence Index across layers, showing 95% Bootstrap Confidence Intervals. (Right) Total Trajectory Length grouped by semantic category.*

Layerwise Similarity Dynamics
=============================

The geometric similarity between adjacent layers reveals the rate of representational change. We consistently observe initial rapid transformation, followed by a stabilization phase, and a final recalibration before the output head.

.. figure:: ../figures/layerwise_similarity.png
   :align: center
   :alt: Layerwise Cosine Similarity
   :width: 600px

   **Figure 5:** Cosine similarity between adjacent layers, illustrating the phase transitions of information processing within the network.