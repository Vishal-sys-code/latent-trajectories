import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import sys

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trajectories import HiddenStateTrajectory

def process_model(model_name: str, hidden_states_dir: str, trajectories_dir: str):
    """
    Processes all hidden state files for a given model and builds trajectory objects.
    
    Args:
        model_name: The name of the model (directory name).
        hidden_states_dir: Base directory for hidden states.
        trajectories_dir: Base directory for output trajectories.
    """
    model_input_dir = os.path.join(hidden_states_dir, model_name)
    model_output_dir = os.path.join(trajectories_dir, model_name)
    
    if not os.path.exists(model_input_dir):
        print(f"Warning: Input directory {model_input_dir} does not exist. Skipping.")
        return
        
    os.makedirs(model_output_dir, exist_ok=True)
    
    metadata_path = os.path.join(model_input_dir, "metadata.parquet")
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file not found at {metadata_path}. Will attempt to infer from .pt files.")
        metadata_df = None
    else:
        metadata_df = pd.read_parquet(metadata_path)
        # Set prompt_id as index for fast lookup
        metadata_df = metadata_df.set_index("prompt_id")
        
    # Get all .pt files except metadata if it somehow ended up as .pt
    pt_files = [f for f in os.listdir(model_input_dir) if f.endswith(".pt")]
    
    print(f"Processing {len(pt_files)} prompts for model {model_name}...")
    
    for pt_file in tqdm(pt_files, desc=model_name):
        # Extract prompt ID from filename
        prompt_id_str = pt_file.replace(".pt", "")
        try:
            # The prompt_id is often an integer, but could be string. Try int first.
            prompt_id = int(prompt_id_str)
        except ValueError:
            prompt_id = prompt_id_str
            
        file_path = os.path.join(model_input_dir, pt_file)
        
        # Load the raw extracted hidden states
        data = torch.load(file_path, weights_only=False)
        
        # Get metadata for this prompt if available
        prompt_text = ""
        model_family = "unknown"
        if metadata_df is not None and prompt_id in metadata_df.index:
            row = metadata_df.loc[prompt_id]
            # Handle potential Series if duplicate index exists
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            prompt_text = row.get("prompt_text", "")
            
        # If it's a dict with "hidden_states", extract it
        if isinstance(data, dict) and "hidden_states" in data:
            hidden_states = data["hidden_states"]
        else:
            # Fallback assuming data is the tensor itself
            hidden_states = data
            
        # The hidden-state tensor is H \in R^{L \times T \times D}
        # We define the trajectory state at layer l as the final-token representation: h_p^{(l)} = H_p^{(l)}[T-1]
        
        # Slicing the final token representation:
        # shape is expected to be [L, T, D], so we do [:, -1, :]
        last_token_states = hidden_states[:, -1, :]
        
        # The first layer is the embedding state (L=0), the rest are transformer blocks (L=1 to N)
        embedding_state = last_token_states[0]
        trajectory_states = last_token_states[1:]
        
        # Create trajectory object
        trajectory = HiddenStateTrajectory(
            prompt_id=prompt_id,
            prompt=prompt_text,
            model=model_name,
            embedding_state=embedding_state,
            trajectory=trajectory_states,
            model_family=model_family
        )
        
        # Save to the standardized format
        out_path = os.path.join(model_output_dir, f"prompt_{prompt_id_str}.pt")
        trajectory.save(out_path)
        
    print(f"Finished processing {model_name}. Output saved to {model_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Build and standardize latent trajectories from hidden states.")
    parser.add_argument("--model", type=str, default=None, help="Specific model to process. If not provided, processes all models.")
    parser.add_argument("--input_dir", type=str, default="data/hidden_states", help="Directory containing raw hidden states.")
    parser.add_argument("--output_dir", type=str, default="data/trajectories", help="Directory to save standard trajectory files.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return
        
    if args.model:
        models_to_process = [args.model]
    else:
        # Get all subdirectories in input_dir
        models_to_process = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
        
    if not models_to_process:
        print(f"No models found in {args.input_dir}.")
        return
        
    print(f"Found {len(models_to_process)} model(s) to process: {', '.join(models_to_process)}")
    
    for model_name in models_to_process:
        process_model(model_name, args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()