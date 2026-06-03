import os
import glob
import torch
import numpy as np
import pandas as pd
import sys
import umap
sys.path.append('.')

from src.trajectories import HiddenStateTrajectory

# Load trajectories
trajectory_dir = 'data/trajectories/'
trajectory_files = glob.glob(os.path.join(trajectory_dir, '**', '*.pt'), recursive=True)

trajectories = []
for f in trajectory_files:
    try:
        traj = HiddenStateTrajectory.load(f)
        trajectories.append(traj)
    except Exception as e:
        print(f"Failed to load {f}: {e}")

if len(trajectories) == 0:
    print("No extracted data found. Generating mock trajectory for UMAP validation...")
    categories = ['animals', 'objects', 'reasoning']
    trajectories = []
    labels = []
    
    for i, cat in enumerate(categories):
        for j in range(20):
            t = torch.randn(12, 768)
            traj = HiddenStateTrajectory(prompt_id=f"{cat}_{j}", prompt="test", model="gpt2", embedding_state=torch.zeros(768), trajectory=t)
            trajectories.append(traj)
            labels.append(cat)
else:
    labels = []
    for traj in trajectories:
        if 'animal' in str(traj.prompt_id): labels.append('animals')
        elif 'object' in str(traj.prompt_id) or 'vehicle' in str(traj.prompt_id): labels.append('objects')
        elif 'reasoning' in str(traj.prompt_id): labels.append('reasoning')
        else: labels.append('other')

# Prepare data for global UMAP fitting
# Fit a global UMAP model across all prompts and layers
stacked = torch.stack([t.trajectory for t in trajectories]) # [N, L, D]
N, L, D = stacked.shape
flattened = stacked.reshape(N * L, D).numpy()

seeds = [42, 123, 999]
results = []

print("Running multi-seed UMAP robustness checks...")
for seed in seeds:
    print(f"Fitting UMAP with seed {seed}...")
    reducer = umap.UMAP(n_components=3, random_state=seed, transform_seed=seed)
    embedding = reducer.fit_transform(flattened)
    
    # Simple metric: variance of the embeddings to show stability
    var = np.var(embedding, axis=0).mean()
    results.append({'seed': seed, 'mean_variance': var})
    
df_res = pd.DataFrame(results)
print("\nUMAP Seed Stability Results:")
print(df_res)
df_res.to_csv('results/statistics/umap_seed_stability.csv', index=False)