import os
import sys
import pytest

sys.path.append(os.path.abspath('src'))
import load_prompts

def test_load_prompts():
    data = load_prompts.load_prompts('data/prompts/prompts.jsonl')
    assert len(data) == 170
    assert all(isinstance(p, dict) for p in data)
    assert all(k in data[0] for k in ['id', 'group', 'subcategory', 'prompt_type', 'difficulty', 'prompt'])

def test_load_dataframe():
    df = load_prompts.load_dataframe('data/prompts/prompts.jsonl')
    assert len(df) == 170
    assert 'group' in df.columns

def test_filters():
    data = load_prompts.load_prompts('data/prompts/prompts.jsonl')
    animals = load_prompts.filter_by_group(data, 'animals')
    assert len(animals) == 50
    assert all(p['group'] == 'animals' for p in animals)

    atomic = load_prompts.filter_by_prompt_type(data, 'atomic')
    assert len(atomic) == 84
    assert all(p['prompt_type'] == 'atomic' for p in atomic)
