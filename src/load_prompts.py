import json

def load_prompts(filepath: str = "data/prompts/prompts.jsonl") -> list[dict]:
    """
    Loads prompts from a JSONL file.
    
    Args:
        filepath (str): Path to the prompts.jsonl file.
        
    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a prompt record.
    """
    prompts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts

def load_dataframe(filepath: str = "data/prompts/prompts.jsonl"):
    """
    Loads prompts into a pandas DataFrame.
    
    Args:
        filepath (str): Path to the prompts.jsonl file.
        
    Returns:
        pd.DataFrame: A DataFrame containing the prompt records.
    """
    import pandas as pd
    prompts = load_prompts(filepath)
    return pd.DataFrame(prompts)

def filter_by_group(data: list[dict], group: str) -> list[dict]:
    """
    Filters a list of prompt dictionaries by group.
    
    Args:
        data (list[dict]): The list of prompt dictionaries.
        group (str): The group to filter by (e.g., 'animals', 'reasoning').
        
    Returns:
        list[dict]: The filtered list of prompts.
    """
    return [p for p in data if p.get("group") == group]

def filter_by_prompt_type(data: list[dict], prompt_type: str) -> list[dict]:
    """
    Filters a list of prompt dictionaries by prompt_type.
    
    Args:
        data (list[dict]): The list of prompt dictionaries.
        prompt_type (str): The prompt_type to filter by (e.g., 'atomic', 'contextual', 'reasoning').
        
    Returns:
        list[dict]: The filtered list of prompts.
    """
    return [p for p in data if p.get("prompt_type") == prompt_type]