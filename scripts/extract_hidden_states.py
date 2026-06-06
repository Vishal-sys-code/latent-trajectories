import os
import argparse
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.load_prompts import load_prompts
from src.trajectories import HiddenStateTrajectory

def run_extraction(
    model_id: str,
    model_name_short: str,
    prompts_file: str = "data/prompts/prompts.jsonl",
    output_base_dir: str = "data/trajectories",
    save_attentions: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Loading model {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="eager" if save_attentions else None
    )
    model.to(device)
    model.eval()

    prompts = load_prompts(prompts_file)
    print(f"Loaded {len(prompts)} prompts.")

    output_dir = os.path.join(output_base_dir, model_name_short)
    os.makedirs(output_dir, exist_ok=True)

    for prompt_data in tqdm(prompts, desc=f"Extracting {model_name_short}"):
        prompt_id = prompt_data["id"]
        prompt_text = prompt_data["prompt"]
        
        # Use 'group' or 'family' depending on what's available
        category = prompt_data.get("family", prompt_data.get("group", "unknown"))

        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=save_attentions)

        # outputs.hidden_states is a tuple of (embedding_layer, layer_1, ..., layer_N)
        stacked_hidden_states = torch.stack(outputs.hidden_states).squeeze(1).cpu().to(torch.float32)

        num_layers, num_tokens, hidden_dim = stacked_hidden_states.shape

        last_token_states = stacked_hidden_states[:, -1, :]  # [num_layers, hidden_dim]

        embedding_state = last_token_states[0]
        trajectory_states = last_token_states[1:]
        
        trajectory = HiddenStateTrajectory(
            prompt_id=prompt_id,
            prompt=prompt_text,
            model=model_name_short,
            embedding_state=embedding_state,
            trajectory=trajectory_states,
            model_family=category
        )
        
        save_path = os.path.join(output_dir, f"{prompt_id}.pt")
        trajectory.save(save_path)

    print(f"Saved {len(prompts)} trajectory files to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, help="Short name of the model to extract (gpt2, tinyllama, qwen2.5)")
    parser.add_argument("--all", action="store_true", help="Extract all three models sequentially")
    args = parser.parse_args()

    models = {
        "gpt2": "gpt2",
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "qwen2.5": "Qwen/Qwen2.5-1.5B"
    }

    if args.all:
        for short_name, full_name in models.items():
            run_extraction(model_id=full_name, model_name_short=short_name)
    elif args.model:
        if args.model.lower() not in models:
            print(f"Unknown model {args.model}. Available models: {', '.join(models.keys())}")
            return
        run_extraction(model_id=models[args.model.lower()], model_name_short=args.model.lower())
    else:
        print("Please specify --model <name> or --all")

if __name__ == "__main__":
    main()