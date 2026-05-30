import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from load_prompts import load_prompts


def run_extraction(
    model_id: str,
    model_name_short: str,
    prompts_file: str = "data/prompts/prompts.jsonl",
    output_base_dir: str = "data/hidden_states",
    save_attentions: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Extracts hidden states from a specified model for all prompts in the dataset.

    Args:
        model_id: HuggingFace model ID (e.g., 'gpt2', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
        model_name_short: Short name for the model directory (e.g., 'gpt2', 'tinyllama')
        prompts_file: Path to prompts JSONL file
        output_base_dir: Base directory to save hidden states
        save_attentions: Whether to also save attention weights
        device: Device to run the model on ('cuda' or 'cpu')
    """
    print(f"Loading model {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Ensure pad token is set to avoid errors during generation/encoding if needed later
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

    metadata = []

    for prompt_data in tqdm(prompts, desc=f"Extracting {model_name_short}"):
        prompt_id = prompt_data["id"]
        prompt_text = prompt_data["prompt"]
        category = prompt_data.get("group", "unknown")

        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(
                **inputs, output_hidden_states=True, output_attentions=save_attentions
            )

        # outputs.hidden_states is a tuple of (embedding_layer, layer_1, ..., layer_N)
        # We want shape [num_layers, num_tokens, hidden_dim]
        # Skip the initial embedding layer to only keep transformer layer outputs, or keep all?
        # Typically people keep all including embeddings, let's keep all.
        # Shape: (num_layers, batch_size, seq_len, hidden_dim) -> (num_layers, seq_len, hidden_dim)
        stacked_hidden_states = torch.stack(outputs.hidden_states).squeeze(1).cpu()

        num_layers, num_tokens, hidden_dim = stacked_hidden_states.shape

        last_token_states = stacked_hidden_states[:, -1, :]  # [num_layers, hidden_dim]

        # Get tokens as strings
        tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]

        save_dict = {
            "input_ids": inputs["input_ids"].cpu(),
            "tokens": tokens,
            "hidden_states": stacked_hidden_states,
            "last_token_states": last_token_states,
            "attention_mask": inputs.get(
                "attention_mask", torch.ones_like(inputs["input_ids"])
            ).cpu(),
        }

        if (
            save_attentions
            and outputs.attentions is not None
            and len(outputs.attentions) > 0
        ):
            # attentions shape per layer: [batch_size, num_heads, seq_len, seq_len]
            stacked_attentions = torch.stack(outputs.attentions).squeeze(1).cpu()
            save_dict["attentions"] = stacked_attentions

        save_path = os.path.join(output_dir, f"{prompt_id}.pt")
        torch.save(save_dict, save_path)

        metadata.append(
            {
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "category": category,
                "model": model_name_short,
                "num_layers": num_layers,
                "num_tokens": num_tokens,
                "hidden_dim": hidden_dim,
            }
        )

    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(output_dir, "metadata.parquet")
    metadata_df.to_parquet(metadata_path, index=False)
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved {len(prompts)} hidden state files to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, default="gpt2", help="HuggingFace model ID"
    )
    parser.add_argument(
        "--model_name_short",
        type=str,
        default="gpt2",
        help="Short name for the model directory",
    )
    parser.add_argument(
        "--prompts_file", type=str, default="data/prompts/prompts.jsonl"
    )
    parser.add_argument("--save_attentions", action="store_true")
    args = parser.parse_args()

    run_extraction(
        model_id=args.model_id,
        model_name_short=args.model_name_short,
        prompts_file=args.prompts_file,
        save_attentions=args.save_attentions,
    )
