import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import numpy as np
import argparse
import pandas as pd
from glob import glob
import joblib

from load_prompts import load_prompts
from dimensionality_reduction import load_all_trajectories
from controls import (
    permute_labels,
    gaussian_embeddings,
    shuffle_embeddings,
    temporal_shuffle,
    run_umap_seed_sweep,
    compare_dr_methods
)

def run_controls_for_model(model_name: str, args):
    print(f"\n--- Running Controls for {model_name} ---")
    
    # Paths
    trajectories_dir = args.trajectories_dir
    prompts_file = args.prompts_file
    models_dir = args.models_dir
    results_dir = args.results_dir
    
    # Load data
    print("Loading trajectories and prompts...")
    trajectories = load_all_trajectories(model_name, trajectories_dir)
    if not trajectories:
        print(f"No trajectories found for {model_name}. Skipping.")
        return
        
    prompts_data = {str(p["id"]): p for p in load_prompts(prompts_file)}
    
    # Extract labels (subcategories) for the trajectories
    labels = []
    valid_trajectories = []
    
    for t in trajectories:
        pid = str(t.prompt_id)
        if pid in prompts_data:
            labels.append(prompts_data[pid].get("subcategory", "unknown"))
            valid_trajectories.append(t)
            
    if not valid_trajectories:
        print("No valid trajectories matched with prompts.")
        return
        
    trajectories = valid_trajectories
    print(f"Loaded {len(trajectories)} trajectories with labels.")
    
    # Control A: Random Labels
    print("Running Control A (Random Label Permutation)...")
    null_dist, real_scores = permute_labels(trajectories, labels, num_permutations=1000)
    
    # Save Control A results
    control_a_data = []
    for layer_idx, real_score in enumerate(real_scores):
        layer_null = null_dist[:, layer_idx]
        p_val = (np.sum(layer_null >= real_score) + 1) / (len(layer_null) + 1)
        z_score = (real_score - np.mean(layer_null)) / (np.std(layer_null) + 1e-9)
        
        control_a_data.append({
            "model": model_name,
            "layer": layer_idx + 1,
            "real_score": real_score,
            "null_mean": np.mean(layer_null),
            "null_std": np.std(layer_null),
            "p_value": p_val,
            "z_score": z_score
        })
    
    pd.DataFrame(control_a_data).to_csv(os.path.join(results_dir, f"{model_name}_label_shuffle.csv"), index=False)
    
    # Save modified trajectories
    print("Generating modified trajectories (Control B1, B2, Phase 7.5)...")
    
    # B1: Gaussian
    gaussian_trajs = gaussian_embeddings(trajectories)
    out_b1 = os.path.join("data", "trajectories_gaussian", model_name)
    os.makedirs(out_b1, exist_ok=True)
    for t in gaussian_trajs:
        t.save(os.path.join(out_b1, f"prompt_{t.prompt_id}.pt"))
        
    # B2: Shuffled
    shuffled_trajs = shuffle_embeddings(trajectories)
    out_b2 = os.path.join("data", "trajectories_shuffled", model_name)
    os.makedirs(out_b2, exist_ok=True)
    for t in shuffled_trajs:
        t.save(os.path.join(out_b2, f"prompt_{t.prompt_id}.pt"))
        
    # Phase 7.5: Temporal Shuffle
    temporal_trajs = temporal_shuffle(trajectories)
    out_75 = os.path.join("data", "trajectories_temporal_shuffle", model_name)
    os.makedirs(out_75, exist_ok=True)
    for t in temporal_trajs:
        t.save(os.path.join(out_75, f"prompt_{t.prompt_id}.pt"))
        
    # Load PCA model for C and D
    pca_path = os.path.join(models_dir, f"{model_name}_pca.pkl")
    if os.path.exists(pca_path):
        print("Loading PCA model...")
        pca_model = joblib.load(pca_path)
        
        # Control C: UMAP seed sweep
        print("Running Control C (UMAP Seed Sweep)...")
        seed_df = run_umap_seed_sweep(trajectories, labels, pca_model)
        seed_df["model"] = model_name
        seed_df.to_csv(os.path.join(results_dir, f"{model_name}_seed_sweep.csv"), index=False)
        
        # Control D: DR comparison
        print("Running Control D (DR Comparison)...")
        dr_df = compare_dr_methods(trajectories, labels, pca_model)
        dr_df["model"] = model_name
        dr_df.to_csv(os.path.join(results_dir, f"{model_name}_dr_comparison.csv"), index=False)
    else:
        print(f"PCA model not found at {pca_path}. Skipping Controls C and D.")
        
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Run controls for representation trajectories.")
    parser.add_argument("--model", type=str, default="all", help="Specific model to run controls for, or 'all'")
    parser.add_argument("--trajectories_dir", type=str, default="data/trajectories", help="Path to trajectories")
    parser.add_argument("--prompts_file", type=str, default="data/prompts/prompts.jsonl", help="Path to prompts")
    parser.add_argument("--models_dir", type=str, default="results/models", help="Path to fitted PCA models")
    parser.add_argument("--results_dir", type=str, default="results/controls", help="Path to save control results")
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    
    if args.model == "all":
        # Find all model directories in trajectories_dir
        if not os.path.exists(args.trajectories_dir):
            print(f"Trajectories dir {args.trajectories_dir} not found.")
            return
            
        model_dirs = [d for d in os.listdir(args.trajectories_dir) 
                      if os.path.isdir(os.path.join(args.trajectories_dir, d))]
                      
        if not model_dirs:
            print("No models found. Please generate trajectories first.")
            return
            
        for model in model_dirs:
            run_controls_for_model(model, args)
    else:
        run_controls_for_model(args.model, args)

if __name__ == "__main__":
    main()