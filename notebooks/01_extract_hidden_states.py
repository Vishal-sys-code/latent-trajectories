# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Phase 3: Hidden State Extraction
# 
# This notebook runs the extraction of transformer hidden states for multiple models using the generated prompts dataset.

# +
import sys
import os
sys.path.append(os.path.abspath(".."))

from src.extract_hidden_states import run_extraction
# -

# ## Run Extraction for GPT-2 Small
# Baseline model for rapid iteration.

# +
run_extraction(
    model_id="gpt2",
    model_name_short="gpt2",
    prompts_file="../data/prompts/prompts.jsonl",
    output_base_dir="../data/hidden_states",
    save_attentions=True
)
# -

# ## Run Extraction for TinyLlama 1.1B
# Primary model for detailed layerwise analysis.

# +
run_extraction(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_name_short="tinyllama",
    prompts_file="../data/prompts/prompts.jsonl",
    output_base_dir="../data/hidden_states",
    save_attentions=True
)
# -

# ## Run Extraction for Qwen 2.5 1.5B
# Comparison model to test generalizability across architectures.

# +
run_extraction(
    model_id="Qwen/Qwen2.5-1.5B",
    model_name_short="qwen",
    prompts_file="../data/prompts/prompts.jsonl",
    output_base_dir="../data/hidden_states",
    save_attentions=True
)
# -

# ## Validation
# Let's perform a sanity check on the GPT-2 outputs to ensure they are shaped correctly and can be loaded.

# +
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import glob

print("--- Validation: GPT-2 ---")

# Load metadata
gpt2_metadata = pd.read_parquet("../data/hidden_states/gpt2/metadata.parquet")
print(f"Total Prompts Extracted: {len(gpt2_metadata)}")
print(f"Model: {gpt2_metadata.iloc[0]['model']}")
print(f"Layers: {gpt2_metadata.iloc[0]['num_layers']}")

pt_files = glob.glob("../data/hidden_states/gpt2/*.pt")
print(f"Files Saved: {len(pt_files)}")

# Select 5 random files
sample_files = random.sample(pt_files, 5)

for i, f in enumerate(sample_files):
    data = torch.load(f)
    print(f"\nSample {i+1} File: {f}")
    print(f"Tokens: {data['tokens']}")
    print(f"Hidden States Shape: {data['hidden_states'].shape}")
    print(f"Last Token States Shape: {data['last_token_states'].shape}")
    if 'attentions' in data:
        print(f"Attentions Shape: {data['attentions'].shape}")

# Simple PCA plot
# Let's take all last_token_states for all prompts for layers 1, 6, 12
layer_idx = [0, 5, 11] # 0-indexed: layer 1, 6, 12
all_data = []

for f in pt_files:
    data = torch.load(f)
    prompt_id = os.path.basename(f).split(".")[0]
    meta = gpt2_metadata[gpt2_metadata["prompt_id"] == prompt_id].iloc[0]
    category = meta["category"]
    
    last_states = data['last_token_states'] # [num_layers, hidden_dim]
    for l_idx in layer_idx:
        all_data.append({
            "layer": l_idx + 1,
            "category": category,
            "state": last_states[l_idx].numpy()
        })

df = pd.DataFrame(all_data)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, l in enumerate([1, 6, 12]):
    ax = axes[i]
    layer_df = df[df["layer"] == l]
    if len(layer_df) == 0:
        continue
        
    X = torch.tensor(layer_df["state"].tolist())
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    layer_df["pca1"] = X_pca[:, 0]
    layer_df["pca2"] = X_pca[:, 1]
    
    categories = layer_df["category"].unique()
    for cat in categories:
        cat_df = layer_df[layer_df["category"] == cat]
        ax.scatter(cat_df["pca1"], cat_df["pca2"], label=cat)
        
    ax.set_title(f"Layer {l} PCA")
    if i == 0:
        ax.legend()
        
plt.tight_layout()
plt.savefig("../data/hidden_states/gpt2_pca_validation.png")
print("\nValidation complete! PCA plot saved to data/hidden_states/gpt2_pca_validation.png")
# -
