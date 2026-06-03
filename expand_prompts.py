import json

def read_existing_prompts(filepath):
    prompts = []
    with open(filepath, 'r') as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts

def write_prompts(filepath, prompts):
    with open(filepath, 'w') as f:
        for p in prompts:
            f.write(json.dumps(p) + '\n')

existing = read_existing_prompts('data/prompts/prompts.jsonl')
print("Existing prompts count:", len(existing))

categories_count = {}
for p in existing:
    categories_count[p['group']] = categories_count.get(p['group'], 0) + 1

print("Counts per group:", categories_count)
