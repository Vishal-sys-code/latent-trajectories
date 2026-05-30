# Trajectory Geometry of Transformer Representations Across Layers

This repository contains the code and experiments for analyzing how hidden representations evolve as structured trajectories through latent space across transformer layers.

## Phase 0 
*   **[Research Specification](docs/research_spec.md)**: Outlines the motivation, hypothesis, models, metrics, and controls.
*   **[Experiment Matrix](docs/experiment_matrix.md)**: Systematic set of planned experiments.

## Phase 2: Prompt Dataset
The prompt dataset provides controlled inputs designed to study latent trajectory evolution across models. It contains a diverse set of examples (Animals, Vehicles, Emotions, Reasoning, and Analogies) varying by `prompt_type` (atomic, contextual, reasoning) and complexity (`difficulty`).

- **Dataset Path:** `data/prompts/prompts.jsonl`
- **Loader Script:** `src/load_prompts.py`
- **Statistics Notebook:** `notebooks/00_dataset_statistics.ipynb`

## Repository Structure
*   `docs/`: Research specifications and documentation.
*   `src/`: Core Python modules for models and metrics.
*   `data/`: Datasets and prompt families.
*   `notebooks/`: Jupyter notebooks for EDA and visualization.
*   `scripts/`: Bash and Python scripts for running experiments.
*   `tests/`: Unit tests for the core modules.