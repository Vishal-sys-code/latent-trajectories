import os
import json
import torch
import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
import umap
from glob import glob
from src.trajectories import HiddenStateTrajectory
from src.load_prompts import load_prompts

def load_all_trajectories(model_name: str, trajectories_dir: str = "data/trajectories"):
    """
    Loads all .pt trajectory files for a given model.
    Returns a list of HiddenStateTrajectory objects.
    """
    model_dir = os.path.join(trajectories_dir, model_name)
    pt_files = glob(os.path.join(model_dir, "*.pt"))
    trajectories = []
    for f in pt_files:
        if "metadata" not in f: # safety check
            traj = HiddenStateTrajectory.load(f)
            trajectories.append(traj)
    return trajectories

def fit_global_models(model_name: str, trajectories_dir: str = "data/trajectories", models_dir: str = "results/models", diagnostics_dir: str = "results/diagnostics"):
    """
    Fits a global PCA(50) and UMAP(3) model sequentially on all hidden states for a given model.
    Saves the models and their diagnostics.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(diagnostics_dir, exist_ok=True)
    
    print(f"Loading trajectories for {model_name}...")
    trajectories = load_all_trajectories(model_name, trajectories_dir)
    
    if not trajectories:
        print(f"No trajectories found for {model_name}.")
        return None, None
        
    print(f"Loaded {len(trajectories)} trajectories.")
    
    # Extract all hidden states (excluding embeddings to focus on layer representations, 
    # but the prompt requires ALL hidden states, so we will use the full trajectory)
    all_states = []
    for t in trajectories:
        # t.trajectory has shape [L, D]
        all_states.append(t.trajectory.numpy())
        
    X = np.concatenate(all_states, axis=0) # [Total_L, D]
    print(f"Stacked states shape: {X.shape}")
    
    # 1. Fit PCA(50)
    print("Fitting PCA(50)...")
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0]))
    X_pca = pca.fit_transform(X)
    
    pca_model_path = os.path.join(models_dir, f"{model_name}_pca.pkl")
    joblib.dump(pca, pca_model_path)
    
    pca_variance = pca.explained_variance_ratio_.tolist()
    pca_diag_path = os.path.join(diagnostics_dir, f"{model_name}_pca_variance.json")
    with open(pca_diag_path, "w") as f:
        json.dump({"cumulative_variance": np.cumsum(pca_variance).tolist(), "variance_ratio": pca_variance}, f)
        
    # 2. Fit UMAP(3) on PCA(50) output
    print("Fitting UMAP(3)...")
    umap_model = umap.UMAP(n_components=3, random_state=42)
    X_umap = umap_model.fit_transform(X_pca)
    
    umap_model_path = os.path.join(models_dir, f"{model_name}_umap.pkl")
    joblib.dump(umap_model, umap_model_path)
    
    # Calculate trustworthiness
    print("Calculating UMAP trustworthiness...")
    # Trustworthiness can be slow for large N, we might sample if N > 10000
    if X_pca.shape[0] > 10000:
        idx = np.random.choice(X_pca.shape[0], 10000, replace=False)
        tw = trustworthiness(X_pca[idx], X_umap[idx])
    else:
        tw = trustworthiness(X_pca, X_umap)
        
    umap_diag_path = os.path.join(diagnostics_dir, f"{model_name}_umap_trustworthiness.json")
    with open(umap_diag_path, "w") as f:
        json.dump({"trustworthiness": float(tw)}, f)
        
    print(f"Models fitted and saved. UMAP Trustworthiness: {tw:.4f}")
    return pca, umap_model


def project_and_save(model_name: str, pca, umap_model, prompts_file: str = "data/prompts/prompts.jsonl", trajectories_dir: str = "data/trajectories", projected_dir: str = "results/projected"):
    """
    Projects all trajectories using the fitted PCA and UMAP models, and saves the coordinates
    along with full metadata to Parquet files.
    """
    os.makedirs(projected_dir, exist_ok=True)
    
    print(f"Projecting trajectories for {model_name}...")
    trajectories = load_all_trajectories(model_name, trajectories_dir)
    prompts_data = {p["id"]: p for p in load_prompts(prompts_file)}
    
    pca_records = []
    umap_records = []
    
    for t in trajectories:
        prompt_id = t.prompt_id
        
        # Handle type mismatch if prompt_id was parsed as int but is string in jsonl
        if str(prompt_id) in prompts_data:
            p_data = prompts_data[str(prompt_id)]
        elif isinstance(prompt_id, int):
            # Attempt to find it by scanning, or just default to empty
            p_data = prompts_data.get(str(prompt_id), {})
            if not p_data:
                p_data = prompts_data.get(prompt_id, {})
        else:
             p_data = prompts_data.get(prompt_id, {})
             
        group = p_data.get("group", "unknown")
        subcategory = p_data.get("subcategory", "unknown")
        
        traj_states = t.trajectory.numpy() # [L, D]
        
        # We need PCA(3) for the PCA visualizations, but the model we saved is PCA(50).
        # We can just take the first 3 components of the PCA(50) projection!
        traj_pca_50 = pca.transform(traj_states)
        traj_pca_3 = traj_pca_50[:, :3]
        
        traj_umap_3 = umap_model.transform(traj_pca_50)
        
        for layer_idx in range(traj_states.shape[0]):
            base_record = {
                "model": model_name,
                "prompt_id": prompt_id,
                "group": group,
                "subcategory": subcategory,
                "layer": layer_idx + 1, # +1 since embedding is 0, these are transformer blocks 1..N
            }
            
            # PCA record
            pca_rec = base_record.copy()
            pca_rec["x"] = traj_pca_3[layer_idx, 0]
            pca_rec["y"] = traj_pca_3[layer_idx, 1]
            pca_rec["z"] = traj_pca_3[layer_idx, 2]
            pca_records.append(pca_rec)
            
            # UMAP record
            umap_rec = base_record.copy()
            umap_rec["x"] = traj_umap_3[layer_idx, 0]
            umap_rec["y"] = traj_umap_3[layer_idx, 1]
            umap_rec["z"] = traj_umap_3[layer_idx, 2]
            umap_records.append(umap_rec)
            
    pca_df = pd.DataFrame(pca_records)
    umap_df = pd.DataFrame(umap_records)
    
    pca_out = os.path.join(projected_dir, f"{model_name}_pca.parquet")
    umap_out = os.path.join(projected_dir, f"{model_name}_umap.parquet")
    
    pca_df.to_parquet(pca_out, index=False)
    umap_df.to_parquet(umap_out, index=False)
    
    print(f"Saved projections to {pca_out} and {umap_out}")
    return pca_df, umap_df