import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().resolve()))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch
from glob import glob

def run_diagnostics():
    from src.dimensionality_reduction import load_all_trajectories
    from src.load_prompts import load_prompts
    
    print("Loading prompts...")
    prompts = load_prompts("data/prompts/prompts.jsonl")
    prompt_dict = {str(p["id"]): p for p in prompts}
    
    print("Loading trajectories...")
    trajectories = load_all_trajectories("gpt2", "data/trajectories")
    
    if not trajectories:
        print("No trajectories found!")
        return
    
    # 1. Print Category Counts
    category_counts = {}
    for t in trajectories:
        pid = str(t.prompt_id)
        if pid in prompt_dict:
            cat = prompt_dict[pid].get("group", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
    print("\n--- Diagnostic A: Category Counts ---")
    for cat, count in category_counts.items():
        print(f"{cat}: {count}")
        
    # 2. Extract final layer embeddings and labels
    final_embeddings = []
    labels = []
    
    for t in trajectories:
        pid = str(t.prompt_id)
        if pid in prompt_dict:
            cat = prompt_dict[pid].get("group", "unknown")
            # Final layer is the last element in trajectory [L, D]
            final_embeddings.append(t.trajectory[-1].numpy())
            labels.append(cat)
            
    final_embeddings = np.array(final_embeddings)
    
    print("\n--- Diagnostic B: PCA 2D Separation ---")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(final_embeddings)
    
    df = pd.DataFrame({
        "PC1": pca_result[:, 0],
        "PC2": pca_result[:, 1],
        "Category": labels
    })
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Category", palette="tab10")
    plt.title("PCA of Final Layer Embeddings")
    plt.savefig("results/diagnostics/diagnostic_pca.png")
    print("Saved PCA plot to results/diagnostics/diagnostic_pca.png")
    
    print("\n--- Diagnostic C: Raw-Space Silhouette Score ---")
    if len(set(labels)) > 1:
        sil_score = silhouette_score(final_embeddings, labels)
        print(f"Silhouette Score (Raw 768-D space): {sil_score:.4f}")
    else:
        print("Not enough categories to compute silhouette score.")

if __name__ == "__main__":
    run_diagnostics()