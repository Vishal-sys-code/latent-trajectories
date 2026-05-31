import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import silhouette_score
import scipy.stats
from trajectories import HiddenStateTrajectory

def compute_trajectory_length(trajectories: List[HiddenStateTrajectory]) -> np.ndarray:
    """
    Computes the total length of the trajectory for each prompt.
    Trajectory length measures how far a representation travels through latent space as it passes through the transformer.
    
    L(T) = Σ || h_(l+1) - h_l ||
    
    Args:
        trajectories: A list of HiddenStateTrajectory objects.
        
    Returns:
        np.ndarray of shape [num_prompts] containing the trajectory lengths.
    """
    lengths = []
    for traj in trajectories:
        dist = traj.layer_distance()
        lengths.append(np.sum(dist) if len(dist) > 0 else 0.0)
    return np.array(lengths)

def compute_curvature(trajectories: List[HiddenStateTrajectory]) -> np.ndarray:
    """
    Computes the average curvature of the trajectory for each prompt.
    Curvature quantifies how much the trajectory bends during processing.
    
    v_l = h_l - h_(l-1)
    v_(l+1) = h_(l+1) - h_l
    κ_l = arccos( (v_l · v_(l+1)) / (||v_l|| ||v_(l+1)||) )
    κ(T) = (1/(L-2)) Σ κ_l
    
    Args:
        trajectories: A list of HiddenStateTrajectory objects.
        
    Returns:
        np.ndarray of shape [num_prompts] containing the overall trajectory curvature.
    """
    curvatures = []
    for traj in trajectories:
        if traj.num_layers < 3:
            curvatures.append(0.0)
            continue
            
        t = traj.trajectory # shape [L, D]
        v = t[1:] - t[:-1]  # shape [L-1, D]
        
        # Calculate cosine similarity between consecutive velocity vectors
        cos_sim = F.cosine_similarity(v[:-1], v[1:], dim=1)
        
        # Ensure values are within valid range for arccos to avoid NaNs due to float imprecision
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        
        # Calculate angles
        angles = torch.acos(cos_sim)
        
        # Calculate mean curvature
        mean_curvature = torch.mean(angles).item()
        curvatures.append(mean_curvature)
        
    return np.array(curvatures)

def compute_layer_velocity(trajectories: List[HiddenStateTrajectory]) -> np.ndarray:
    """
    Computes the average velocity at each layer transition across all prompts.
    Velocity measures the amount of movement between consecutive layers.
    
    v_l = || h_(l+1) - h_l ||
    
    Args:
        trajectories: A list of HiddenStateTrajectory objects.
        
    Returns:
        np.ndarray of shape [L-1] containing the mean velocity at each layer transition.
    """
    if not trajectories:
        return np.array([])
        
    num_layers = trajectories[0].num_layers
    if num_layers < 2:
        return np.array([])
        
    # Stack layer distances to compute mean across prompts
    all_distances = []
    for traj in trajectories:
        all_distances.append(traj.layer_distance())
        
    velocities = np.stack(all_distances) # [num_prompts, L-1]
    return np.mean(velocities, axis=0) # [L-1]

def compute_convergence_matrix(trajectories: List[HiddenStateTrajectory]) -> np.ndarray:
    """
    Computes a pairwise convergence matrix for a list of trajectories.
    Positive values indicate convergence, negative values indicate divergence.
    
    D_ij(l) = || T_i(l) - T_j(l) ||
    C_ij = D_ij(1) - D_ij(L)
    
    Args:
        trajectories: A list of HiddenStateTrajectory objects.
        
    Returns:
        np.ndarray of shape [num_prompts, num_prompts] containing pairwise convergence scores.
    """
    n = len(trajectories)
    if n == 0:
        return np.array([])
        
    convergence_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            traj_i = trajectories[i].trajectory
            traj_j = trajectories[j].trajectory
            
            # Distance at layer 1 (index 0)
            d_init = torch.norm(traj_i[0] - traj_j[0], p=2).item()
            
            # Distance at layer L (index -1)
            d_final = torch.norm(traj_i[-1] - traj_j[-1], p=2).item()
            
            c_ij = d_init - d_final
            convergence_matrix[i, j] = c_ij
            convergence_matrix[j, i] = c_ij
            
    return convergence_matrix

def compute_layerwise_silhouette(trajectories: List[HiddenStateTrajectory], labels: List[str]) -> np.ndarray:
    """
    Computes the silhouette score at each layer to determine category separation.
    
    Args:
        trajectories: A list of HiddenStateTrajectory objects.
        labels: A list of category labels for each prompt.
        
    Returns:
        np.ndarray of shape [L] containing the silhouette score for each layer.
    """
    if not trajectories or not labels:
        return np.array([])
        
    num_layers = trajectories[0].num_layers
    n = len(trajectories)
    
    # Check if we have at least 2 categories and more than 2 samples
    unique_labels = set(labels)
    if len(unique_labels) < 2 or n < 2:
        return np.zeros(num_layers)
        
    scores = []
    
    for l in range(num_layers):
        # Extract embeddings for all prompts at layer l
        layer_embeddings = torch.stack([traj.trajectory[l] for traj in trajectories]).numpy()
        
        try:
            score = silhouette_score(layer_embeddings, labels)
        except ValueError:
            # Handle edge cases (e.g., only 1 label present in batch)
            score = 0.0
            
        scores.append(score)
        
    return np.array(scores)

def compute_rsa_matrix(trajectories: List[HiddenStateTrajectory]) -> np.ndarray:
    """
    Performs Representational Similarity Analysis (RSA) by computing the 
    Spearman rank correlation between the Representational Dissimilarity Matrix (RDM) 
    of each pair of layers.
    
    Args:
        trajectories: A list of HiddenStateTrajectory objects.
        
    Returns:
        np.ndarray of shape [L, L] containing the RSA matrix.
    """
    if not trajectories:
        return np.array([])
        
    num_layers = trajectories[0].num_layers
    n = len(trajectories)
    
    if n < 2 or num_layers < 2:
        return np.zeros((num_layers, num_layers))
        
    rdms = []
    
    # Pre-compute RDMs for all layers
    for l in range(num_layers):
        layer_embeddings = torch.stack([traj.trajectory[l] for traj in trajectories]) # [N, D]
        
        # Compute normalized cosine similarity matrix
        # Cosine similarity formula: (A @ B.T) / (||A|| ||B||)
        # Using F.cosine_similarity pairwise is O(N^2), but matrix mult is faster
        norm_embeddings = F.normalize(layer_embeddings, p=2, dim=1)
        sim_matrix = torch.mm(norm_embeddings, norm_embeddings.t())
        
        # RDM = 1 - Cosine Similarity
        rdm = 1.0 - sim_matrix
        
        # Extract upper triangle (excluding diagonal) for correlation
        triu_indices = torch.triu_indices(row=n, col=n, offset=1)
        rdm_flat = rdm[triu_indices[0], triu_indices[1]].numpy()
        rdms.append(rdm_flat)
        
    # Compute L x L RSA Matrix
    rsa_matrix = np.zeros((num_layers, num_layers))
    
    for i in range(num_layers):
        for j in range(num_layers):
            if i == j:
                rsa_matrix[i, j] = 1.0
            elif i < j:
                corr, _ = scipy.stats.spearmanr(rdms[i], rdms[j])
                rsa_matrix[i, j] = corr
                rsa_matrix[j, i] = corr
                
    return rsa_matrix