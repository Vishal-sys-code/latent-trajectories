"""Generate F5 (ambiguous) and F5_control (unambiguous matched) prompt pairs for bifurcation analysis.

This script creates pairs of prompts with the same ambiguous word in disambiguating contexts,
plus matched unambiguous controls for comparison. These are used to analyze trajectory bifurcation
as described in Finding 3 of the paper.
"""
import json
from pathlib import Path

# F5 Ambiguous prompts: pairs with same ambiguous word in different contexts
# Each tuple: (id_suffix, ambiguous_word, context1_prompt, context2_prompt)
f5_ambiguous_pairs = [
    ("001", "bank", "The river flows past the green bank.", "She deposited money at the bank."),
    ("002", "bat", "The bat hung from the cave ceiling.", "He swings the bat at the baseball."),
    ("003", "light", "The feather is very light and floats easily.", "The light from the lamp fills the room."),
    ("004", "bass", "The bass swims in the deep lake.", "The bass guitar plays a deep sound."),
    ("005", "spring", "Spring arrives and flowers bloom.", "A metal spring stretches and bounces back."),
    ("006", "date", "The date is written on the calendar.", "They plan a dinner date together."),
    ("007", "bark", "The dog's bark echoes down the street.", "The bark of the tree feels rough."),
    ("008", "match", "The match lit the candle on fire.", "The colors match perfectly together."),
    ("009", "tie", "He wears a tie with his business suit.", "The knot will tie the rope securely."),
    ("010", "present", "She opens the present on her birthday.", "He will present the findings tomorrow."),
    ("011", "can", "She can finish the job by Friday.", "The can holds tomato soup inside."),
    ("012", "wind", "The wind blows through the trees.", "Watch the wind as it tightens the rope."),
    ("013", "tear", "A tear rolls down her cheek.", "Do not tear the paper when opening it."),
    ("014", "bowl", "He fills the bowl with soup.", "She will bowl at the league tonight."),
    ("015", "live", "We live in the city with our family.", "The band will live stream the concert."),
]

# F5_control Unambiguous prompts: matched syntax but unambiguous words
# Each tuple: (id_suffix, control_word, context1_prompt, context2_prompt)
f5_control_pairs = [
    ("001", "shore", "The river flows past the green shore.", "She went to the shore yesterday."),
    ("002", "bird", "The bird hung from the cave ceiling.", "He watches the bird in the garden."),
    ("003", "feather", "The feather is very light and floats easily.", "The feather drifts through the air."),
    ("004", "fish", "The fish swims in the deep lake.", "The fish jumps out of the water."),
    ("005", "season", "Spring season arrives and flowers bloom.", "The season changes with time."),
    ("006", "number", "The number is written on the calendar.", "They choose a lucky number.", ),
    ("007", "sound", "The dog's sound echoes down the street.", "The sound of the tree feels nice."),
    ("008", "game", "The game lit the candle on fire.", "The colors of the game work together."),
    ("009", "knot", "He wears a knot with his business suit.", "The knot will secure the rope safely."),
    ("010", "gift", "She opens the gift on her birthday.", "He will give the findings tomorrow."),
    ("011", "container", "She fills the container with juice inside.", "The container holds things safely."),
    ("012", "breeze", "The breeze blows through the trees.", "Watch the breeze as it flows gently."),
    ("013", "cry", "A cry comes from her mouth.", "Do not cry when opening the box."),
    ("014", "dish", "He fills the dish with soup.", "She will serve the dish at dinner."),
    ("015", "stay", "We stay in the city with family.", "The band will stay for the concert."),
]

def main():
    # Use absolute path based on script location
    script_dir = Path(__file__).parent.parent
    prompts_path = script_dir / 'data' / 'prompts' / 'prompts.jsonl'
    
    # Load existing prompts
    existing = {}
    if prompts_path.exists():
        with open(prompts_path, 'r') as f:
            for line in f:
                p = json.loads(line)
                existing[p['id']] = p
    
    new_prompts = []
    
    # Add F5 ambiguous pairs
    for suffix, word, ctx1, ctx2 in f5_ambiguous_pairs:
        new_prompts.append({
            "id": f"ambiguous_{suffix}",
            "group": "ambiguous",
            "prompt_type": "disambiguation",
            "difficulty": "medium",
            "ambiguous_word": word,
            "context": "ctx1",
            "prompt": ctx1
        })
        new_prompts.append({
            "id": f"ambiguous_{suffix}_b",
            "group": "ambiguous",
            "prompt_type": "disambiguation",
            "difficulty": "medium",
            "ambiguous_word": word,
            "context": "ctx2",
            "prompt": ctx2
        })
    
    # Add F5_control unambiguous pairs
    for suffix, word, ctx1, ctx2 in f5_control_pairs:
        new_prompts.append({
            "id": f"ambiguous_control_{suffix}",
            "group": "ambiguous_control",
            "prompt_type": "disambiguation",
            "difficulty": "medium",
            "control_word": word,
            "context": "ctx1",
            "prompt": ctx1
        })
        new_prompts.append({
            "id": f"ambiguous_control_{suffix}_b",
            "group": "ambiguous_control",
            "prompt_type": "disambiguation",
            "difficulty": "medium",
            "control_word": word,
            "context": "ctx2",
            "prompt": ctx2
        })
    
    # Merge with existing and write back
    all_prompts = list(existing.values()) + new_prompts
    
    with open(prompts_path, 'w') as f:
        for p in all_prompts:
            f.write(json.dumps(p) + '\n')
    
    print(f"Added {len(new_prompts)} F5 and F5_control prompts")
    print(f"Total prompts: {len(all_prompts)}")
    
    # Count by group
    counts = {}
    for p in all_prompts:
        g = p.get('group', 'unknown')
        counts[g] = counts.get(g, 0) + 1
    print("Counts per group:", dict(sorted(counts.items())))

if __name__ == '__main__':
    main()
