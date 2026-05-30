# Research Specification: Trajectory Geometry of Transformer Representations Across Layers

## 1. Motivation
Current analyses of large language models often focus on static embeddings or final outputs. However, viewing representations through a computational lens suggests that transformers build knowledge iteratively, layer by layer. Understanding how hidden representations dynamically traverse the latent space—whether they follow structured geometric paths or random walks—can bridge the gap between mechanistic interpretability and representational learning. By framing transformer computation as a trajectory, we can evaluate how and where structured reasoning, semantic convergence, and disambiguation occur across layers. This structural insight is crucial for understanding the latent neural manifolds and could lay the groundwork for better interventions, pruning, or training strategies.

## 2. Research Question
Do hidden representations evolve as structured trajectories through latent space, rather than remaining as static layerwise embeddings? Specifically, how do semantically related concepts behave geometrically as they pass through successive transformer layers?

## 3. Hypothesis
**Hypothesis H1:** Semantically related prompts exhibit measurable trajectory convergence and structured divergence across transformer layers. As computation progresses, related concepts draw closer in latent space, while structured reasoning tasks form distinct geometric paths.

**Null Hypothesis H0:** Observed trajectory patterns are artifacts of dimensionality reduction and disappear under rigorous controls (such as shuffled layers or random embeddings). There is no inherent layerwise geometric structure.

## 4. Dataset
To ensure rigorous testing and avoid arbitrary results, we employ controlled prompt families rather than random text:

*   **Family 1: Semantic Categories:** E.g., (cat, dog, lion), (car, truck, airplane), (happy, sad, angry). *Purpose: Measure semantic convergence.*
*   **Family 2: Lexical Variations:** E.g., "cat", "a cat", "the cat", "small cat", "large cat". *Purpose: Assess robustness.*
*   **Family 3: Analogies:** E.g., "king queen", "man woman", "paris france". *Purpose: Observe structured reasoning geometry.*
*   **Family 4: Reasoning Prompts:** E.g., "A > B, B > C, therefore?". *Purpose: Track dynamic computation.*
*   **Family 5: Ambiguous Concepts:** E.g., "bank", "bat", "light". *Purpose: Examine representation splitting and disambiguation.*

## 5. Models
To prioritize reproducibility, rapid iteration, and clear comparative baselines over scale, we will analyze the following models:

*   **Model A: GPT-2 Small** - Extremely easy, fast, and reproducible.
*   **Model B: TinyLlama** - A modern architecture that remains small enough for extensive layerwise analysis.
*   **Model C: Qwen 2.5** - A stronger model serving as an advanced comparison point.

*(Note: API-only and 70B+ models are deliberately excluded to ensure full local control and computational feasibility.)*

## 6. Metrics
We move beyond visualization to quantitative metrics for publication:

*   **Metric 1: Trajectory Length ($L$)** - $L = \sum_i ||h_i - h_{i+1}||$. Measures how much the representation evolves across layers.
*   **Metric 2: Trajectory Curvature ($\kappa$)** - Assesses the non-linearity of the path, particularly testing if reasoning prompts induce higher curvature.
*   **Metric 3: Semantic Convergence ($D(t)$)** - The distance between semantic groups across layers. Tests if related concepts converge.
*   **Metric 4: Representational Similarity Analysis (RSA)** - Compares representation geometry across different models or prompt families. A standard in computational neuroscience.
*   **Metric 5: Trajectory Stability** - A novel metric measuring the robustness of a trajectory against lexical perturbations.

## 7. Controls
Controls are crucial to prove that observed structures are not artifacts of the methodology:

*   **Control A: Random Labels** - Compute convergence for randomly grouped words. *Expected: No meaningful convergence.*
*   **Control B: Random Embeddings** - Pass inputs through an untrained network. *Expected: No geometric structure.*
*   **Control C: Shuffled Layers** - Reorder the layer sequence and recompute metrics. *Expected: Trajectory structure disappears, proving layerwise dependency.*
*   **Control D: Multiple Dimensionality Reductions** - Validate findings across PCA, UMAP, and t-SNE. *Expected: Consistent geometry regardless of the reduction algorithm.*

## 8. Expected Findings
1.  **Semantic Clustering:** Words within Semantic Categories will start dispersed in early layers and converge into tight clusters in deeper layers.
2.  **Disambiguation:** Ambiguous Concepts will show a clear trajectory split midway through the network as context dictates meaning.
3.  **Reasoning Non-linearity:** Reasoning and Analogy tasks will exhibit significantly higher Trajectory Curvature compared to Lexical Variations.
4.  **Control Collapse:** All structured geometric properties will vanish under shuffled layers and random embeddings, confirming the structural integrity of the findings.

## 9. Risks
1.  **Dimensionality Reduction Artifacts:** Over-reliance on visual plots like t-SNE can lead to pareidolia (seeing patterns that don't exist). *Mitigation: Strict reliance on high-dimensional Metrics 1-5 and Control D.*
2.  **Model Specificity:** Findings may only hold true for GPT-2 and not generalize to modern architectures. *Mitigation: Using TinyLlama and Qwen 2.5 as counter-balances.*
3.  **Computational Overhead:** Computing pairwise distances for all tokens across all layers might cause memory bottlenecks. *Mitigation: Using small prompt families and small models.*

## 10. Success Criteria (Deliverable D)
We consider the Phase 1 execution of this project successful if:
1.  **Observable Convergence:** Semantic categories show statistically significant measurable convergence across layers.
2.  **Replicability:** Core results replicate across at least 2 out of the 3 chosen models.
3.  **Control Validation:** All controls successfully eliminate the observed trajectory structures (proving H0 can be rejected).
4.  **Publication Viability:** At least one statistically significant finding survives rigorous ablation and is robust enough for a workshop paper submission.
