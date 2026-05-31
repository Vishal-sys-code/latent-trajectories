import numpy as np
import torch
import pandas as pd
from typing import List, Dict, Tuple, Optional
import copy
from src.trajectories import HiddenStateTrajectory
from src.metrics import compute_layerwise_silhouette, compute_trajectory_length, compute_curvature
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def permute_labels(
    trajectories: List[HiddenStateTrajectory], 
    labels: List[str], 
    num_permutations: int = 1000, 
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Control A: Random Label Permutation.
    Destroys semantic structure by shuffling labels while keeping embeddings intact.
    
    Returns:
        null_distribution: shape [num_permutations, L] (silhouette scores for each layer)
        real_scores: shape [L] (silhouette scores with real labels)
    """
    np.random.seed(seed)
    
    # Compute real scores
    real_scores = compute_layerwise_silhouette(trajectories, labels)
    
    # Generate null distribution
    null_distribution = []
    
    labels_array = np.array(labels)
    for _ in range(num_permutations):
        shuffled_labels = np.random.permutation(labels_array).tolist()
        scores = compute_layerwise_silhouette(trajectories, shuffled_labels)
        null_distribution.append(scores)
        
    return np.array(null_distribution), real_scores

def gaussian_embeddings(trajectories: List[HiddenStateTrajectory], seed: int = 42) -> List[HiddenStateTrajectory]:
    """
    Control B1: Gaussian Noise Control.
    Replaces representations with X_random ~ N(mean_real, std_real) per layer.
    """
    torch.manual_seed(seed)
    if not trajectories:
        return []
        
    num_layers = trajectories[0].num_layers
    hidden_dim = trajectories[0].hidden_dim
    n = len(trajectories)
    
    # Stack all trajectories: [N, L, D]
    stacked = torch.stack([t.trajectory for t in trajectories])
    
    # Calculate mean and std per layer: [L, D]
    layer_means = torch.mean(stacked, dim=0)
    layer_stds = torch.std(stacked, dim=0)
    
    # Generate random trajectories
    random_trajectories = []
    for i in range(n):
        # Generate N(0, 1) and scale to match layer stats
        noise = torch.randn((num_layers, hidden_dim))
        random_traj_data = noise * layer_stds + layer_means
        
        # We also replace the embedding state
        emb_mean = torch.mean(torch.stack([t.embedding_state for t in trajectories]), dim=0)
        emb_std = torch.std(torch.stack([t.embedding_state for t in trajectories]), dim=0)
        random_emb = torch.randn(hidden_dim) * emb_std + emb_mean
        
        t_original = trajectories[i]
        
        new_traj = HiddenStateTrajectory(
            prompt_id=t_original.prompt_id,
            prompt=t_original.prompt,
            model=t_original.model,
            embedding_state=random_emb,
            trajectory=random_traj_data,
            model_family=t_original.model_family
        )
        random_trajectories.append(new_traj)
        
    return random_trajectories

def shuffle_embeddings(trajectories: List[HiddenStateTrajectory], seed: int = 42) -> List[HiddenStateTrajectory]:
    """
    Control B2: Layer-wise Shuffled Embeddings.
    Keeps real vectors but destroys correspondence across prompts per layer.
    """
    np.random.seed(seed)
    if not trajectories:
        return []
        
    n = len(trajectories)
    num_layers = trajectories[0].num_layers
    
    # Extract all trajectories into a list of tensors: L x list of N tensors
    layer_tensors = [[t.trajectory[l] for t in trajectories] for l in range(num_layers)]
    
    # Shuffle each layer independently
    shuffled_layer_tensors = []
    for l in range(num_layers):
        indices = np.random.permutation(n)
        shuffled_layer_tensors.append([layer_tensors[l][i] for i in indices])
        
    # Reconstruct trajectories
    shuffled_trajectories = []
    for i in range(n):
        t_original = trajectories[i]
        
        # Stack the shuffled tensors for this prompt
        new_traj_data = torch.stack([shuffled_layer_tensors[l][i] for l in range(num_layers)])
        
        # Shuffle embeddings state as well across all prompts
        all_embs = [t.embedding_state for t in trajectories]
        rand_emb_idx = np.random.randint(n)
        
        new_traj = HiddenStateTrajectory(
            prompt_id=t_original.prompt_id,
            prompt=t_original.prompt,
            model=t_original.model,
            embedding_state=all_embs[rand_emb_idx],
            trajectory=new_traj_data,
            model_family=t_original.model_family
        )
        shuffled_trajectories.append(new_traj)
        
    return shuffled_trajectories

def temporal_shuffle(trajectories: List[HiddenStateTrajectory], seed: int = 42) -> List[HiddenStateTrajectory]:
    """
    Phase 7.5: Temporal Trajectory Destruction.
    Randomly reorders the layer indices for each trajectory independently.
    """
    np.random.seed(seed)
    if not trajectories:
        return []
        
    num_layers = trajectories[0].num_layers
    
    shuffled_trajectories = []
    for i, t in enumerate(trajectories):
        layer_indices = np.random.permutation(num_layers)
        
        new_traj_data = t.trajectory[layer_indices]
        
        new_traj = HiddenStateTrajectory(
            prompt_id=t.prompt_id,
            prompt=t.prompt,
            model=t.model,
            embedding_state=t.embedding_state,
            trajectory=new_traj_data,
            model_family=t.model_family
        )
        shuffled_trajectories.append(new_traj)
        
    return shuffled_trajectories

def run_umap_seed_sweep(trajectories: List[HiddenStateTrajectory], labels: List[str], pca_model, seeds: List[int] = [0, 42, 123, 999]) -> pd.DataFrame:
    """
    Control C: Different Random Seeds for UMAP.
    Verifies that metrics remain stable across UMAP runs.
    """
    if not trajectories or not labels:
        return pd.DataFrame()
        
    # Flatten trajectories
    stacked = np.concatenate([t.trajectory.numpy() for t in trajectories], axis=0) # [N*L, D]
    
    # Project with PCA first
    X_pca = pca_model.transform(stacked)
    
    results = []
    
    # Calculate properties per layer
    num_layers = trajectories[0].num_layers
    n = len(trajectories)
    
    for seed in seeds:
        umap_model = umap.UMAP(n_components=3, random_state=seed)
        X_umap = umap_model.fit_transform(X_pca)
        
        # Reshape to [N, L, 3] to evaluate layerwise clustering
        X_umap_reshaped = X_umap.reshape(n, num_layers, 3)
        
        # Compute silhouette score for last layer as a summary metric
        from sklearn.metrics import silhouette_score
        try:
            sil_score = silhouette_score(X_umap_reshaped[:, -1, :], labels)
        except ValueError:
            sil_score = 0.0
            
        # We can also measure average trajectory length in UMAP space
        lengths = []
        for i in range(n):
            diffs = X_umap_reshaped[i, 1:, :] - X_umap_reshaped[i, :-1, :]
            dists = np.linalg.norm(diffs, axis=1)
            lengths.append(np.sum(dists))
        avg_length = np.mean(lengths)
        
        results.append({
            "seed": seed,
            "final_layer_silhouette": sil_score,
            "avg_trajectory_length": avg_length
        })
        
    return pd.DataFrame(results)

def compare_dr_methods(trajectories: List[HiddenStateTrajectory], labels: List[str], pca_model) -> pd.DataFrame:
    """
    Control D: PCA vs UMAP vs t-SNE.
    Verifies that the core geometry and separation remains robust to DR method.
    """
    if not trajectories or not labels:
        return pd.DataFrame()
        
    # Flatten trajectories
    stacked = np.concatenate([t.trajectory.numpy() for t in trajectories], axis=0) # [N*L, D]
    X_pca_50 = pca_model.transform(stacked)
    
    # We compare 3 methods to reduce from 50 to 3
    # 1. PCA(3) (first 3 components of PCA50)
    X_pca_3 = X_pca_50[:, :3]
    
    # 2. UMAP(3)
    umap_model = umap.UMAP(n_components=3, random_state=42)
    X_umap_3 = umap_model.fit_transform(X_pca_50)
    
    # 3. tSNE(3)
    tsne_model = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto')
    X_tsne_3 = tsne_model.fit_transform(X_pca_50)
    
    results = []
    
    num_layers = trajectories[0].num_layers
    n = len(trajectories)
    
    methods = {
        "PCA": X_pca_3,
        "UMAP": X_umap_3,
        "tSNE": X_tsne_3
    }
    
    from sklearn.metrics import silhouette_score
    
    for method_name, X_proj in methods.items():
        X_reshaped = X_proj.reshape(n, num_layers, 3)
        
        # Metric 1: Separation at last layer
        try:
            sil_score = silhouette_score(X_reshaped[:, -1, :], labels)
        except ValueError:
            sil_score = 0.0
            
        # Metric 2: Average trajectory length
        lengths = []
        for i in range(n):
            diffs = X_reshaped[i, 1:, :] - X_reshaped[i, :-1, :]
            dists = np.linalg.norm(diffs, axis=1)
            lengths.append(np.sum(dists))
        avg_length = np.mean(lengths)
        
        # Metric 3: Convergence (distance init - distance final) normalized
        convergence_scores = []
        for i in range(n):
            for j in range(i + 1, n):
                d_init = np.linalg.norm(X_reshaped[i, 0, :] - X_reshaped[j, 0, :])
                d_final = np.linalg.norm(X_reshaped[i, -1, :] - X_reshaped[j, -1, :])
                convergence_scores.append(d_init - d_final)
                
        avg_convergence = np.mean(convergence_scores) if convergence_scores else 0.0
        
        results.append({
            "method": method_name,
            "final_layer_silhouette": sil_score,
            "avg_trajectory_length": avg_length,
            "avg_convergence": avg_convergence
        })
        
    return pd.DataFrame(results)