import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('.')

from src.trajectories import HiddenStateTrajectory
from src.metrics import compute_per_prompt_convergence_score
from src.stats import compare_distributions, bootstrap_ci, permutation_test, cohens_d

sns.set_theme(style="whitegrid")

# Load trajectories
trajectory_dir = 'data/trajectories/'
trajectory_files = glob.glob(os.path.join(trajectory_dir, '**', '*.pt'), recursive=True)

all_trajectories = []
for f in trajectory_files:
    try:
        traj = HiddenStateTrajectory.load(f)
        all_trajectories.append(traj)
    except Exception as e:
        print(f"Failed to load {f}: {e}")

if len(all_trajectories) == 0:
    print("No extracted data found. Generating mock trajectory for convergence analysis validation...")
    categories = ['animals', 'objects', 'reasoning']
    all_trajectories = []
    
    for i, cat in enumerate(categories):
        for j in range(20):
            t = torch.randn(12, 768)
            # Add some simulated convergence effect based on category
            if cat == 'animals': t += torch.linspace(0, 1, 12).unsqueeze(-1) * 0.5
            elif cat == 'objects': t -= torch.linspace(0, 1, 12).unsqueeze(-1) * 0.5
            elif cat == 'reasoning': t += torch.linspace(0, 2, 12).unsqueeze(-1) * 0.5
                
            traj = HiddenStateTrajectory(prompt_id=f"{cat}_{j}", prompt="test", model="gpt2", embedding_state=torch.zeros(768), trajectory=t)
            all_trajectories.append(traj)

# Group trajectories by model
models = list(set([t.model for t in all_trajectories]))
if not models:
    models = ["mock_model"]

results = []
comparisons = []

# Prepare figure for separate panels
fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6), sharey=True)
if len(models) == 1:
    axes = [axes]

for ax, model_name in zip(axes, models):
    model_trajs = [t for t in all_trajectories if getattr(t, 'model', 'mock_model') == model_name]
    
    labels = []
    for traj in model_trajs:
        if 'animal' in str(traj.prompt_id): labels.append('animals')
        elif 'object' in str(traj.prompt_id) or 'vehicle' in str(traj.prompt_id): labels.append('objects')
        elif 'reasoning' in str(traj.prompt_id): labels.append('reasoning')
        else: labels.append('other')
            
    # filter to only target categories
    valid_indices = [i for i, l in enumerate(labels) if l in ['animals', 'objects', 'reasoning']]
    model_trajs = [model_trajs[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    
    if not model_trajs:
        print(f"No valid trajectories for model {model_name}")
        continue

    print(f"Computing convergence scores for model: {model_name}...")
    scores_per_prompt = compute_per_prompt_convergence_score(model_trajs, labels)
    num_layers = scores_per_prompt.shape[1]
    unique_labels = sorted(list(set(labels)))

    colors = sns.color_palette("husl", len(unique_labels))

    layers = np.arange(1, num_layers + 1)
    x_norm = np.linspace(0, 1, num_layers)
    mean_scores_by_cat = {}
    ci_bounds_by_cat = {}

    for idx, cat in enumerate(unique_labels):
        cat_indices = [i for i, l in enumerate(labels) if l == cat]
        cat_scores = scores_per_prompt[cat_indices]  # [N_cat, L]

        mean_scores = []
        lower_cis = []
        upper_cis = []

        for l in range(num_layers):
            layer_scores = cat_scores[:, l]
            mean_score = np.mean(layer_scores)
            lower_ci, upper_ci = bootstrap_ci(layer_scores, num_bootstraps=1000, confidence_level=0.95, random_seed=42)

            mean_scores.append(mean_score)
            lower_cis.append(lower_ci)
            upper_cis.append(upper_ci)

            results.append({
                'model': model_name,
                'category': cat,
                'layer': l + 1,
                'mean_tci': mean_score,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci
            })

        mean_scores_by_cat[cat] = np.array(mean_scores)
        ci_bounds_by_cat[cat] = (np.array(lower_cis), np.array(upper_cis))

    ax.axhline(0, ls='--', color='gray', alpha=0.7)
    for idx, cat in enumerate(unique_labels):
        mean_scores = mean_scores_by_cat[cat]
        lower_cis, upper_cis = ci_bounds_by_cat[cat]
        ax.plot(x_norm, mean_scores, label=cat, color=colors[idx], linewidth=2.5)
        ax.fill_between(x_norm, lower_cis, upper_cis, color=colors[idx], alpha=0.2)

    ax.set_title(f'{model_name} ({num_layers} layers)', fontsize=16)
    ax.set_xlabel('Normalized Layer Position', fontsize=14)
    ax.set_xticks(x_norm)
    ax.set_xticklabels(layers)
    if ax == axes[0]:
        ax.set_ylabel('Convergence Score ($D_{between} - D_{within}$)', fontsize=14)
    ax.legend()

    # Key Statistical Comparisons (Phase 11) for this model
    overall_tci = np.mean(scores_per_prompt, axis=0) # [L]

    start_layer_scores = scores_per_prompt[:, 0]
    final_layer_scores = scores_per_prompt[:, -1]
    p_val_A = permutation_test(start_layer_scores, final_layer_scores, random_seed=42)
    d_A = cohens_d(start_layer_scores, final_layer_scores)
    comparisons.append({
        'model': model_name,
        'Comparison': 'Start vs End',
        'p_value': p_val_A,
        'cohens_d': d_A
    })

    peak_layer_idx = np.argmax(overall_tci)
    peak_layer_scores = scores_per_prompt[:, peak_layer_idx]
    p_val_B = permutation_test(start_layer_scores, peak_layer_scores, random_seed=42)
    d_B = cohens_d(start_layer_scores, peak_layer_scores)
    comparisons.append({
        'model': model_name,
        'Comparison': 'Start vs Peak',
        'p_value': p_val_B,
        'cohens_d': d_B
    })

    semantic_cats = ['animals', 'objects']
    semantic_indices = [i for i, l in enumerate(labels) if l in semantic_cats]
    reasoning_indices = [i for i, l in enumerate(labels) if l == 'reasoning']
    if len(semantic_indices) > 0 and len(reasoning_indices) > 0:
        semantic_final = scores_per_prompt[semantic_indices, -1]
        reasoning_final = scores_per_prompt[reasoning_indices, -1]
        p_val_C = permutation_test(semantic_final, reasoning_final, random_seed=42)
        d_C = cohens_d(semantic_final, reasoning_final)
        comparisons.append({
            'model': model_name,
            'Comparison': 'Semantic vs Reasoning',
            'p_value': p_val_C,
            'cohens_d': d_C
        })

fig.suptitle('Convergence Score Across Layers', fontsize=15, y=1.05)
plt.tight_layout()

os.makedirs('figures/', exist_ok=True)
plt.savefig('figures/convergence_score_layers.pdf', bbox_inches='tight')
plt.savefig('figures/convergence_score_layers.png', bbox_inches='tight', dpi=300)
print("Saved figures/convergence_score_layers.pdf/png")

df_conv = pd.DataFrame(results)
os.makedirs('results/', exist_ok=True)
df_conv.to_csv('results/convergence_metrics.csv', index=False)
print("Saved results/convergence_metrics.csv")

df_comparisons = pd.DataFrame(comparisons)
os.makedirs('results/statistics/', exist_ok=True)
df_comparisons.to_csv('results/statistics/convergence_summary.csv', index=False)
print("Saved results/statistics/convergence_summary.csv")