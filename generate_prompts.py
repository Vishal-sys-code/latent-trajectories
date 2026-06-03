import json

new_prompts = []
start_id = 11

def generate_id(group, idx):
    prefix = ""
    if group == "animals": prefix = "animal"
    elif group == "objects": prefix = "object"
    elif group == "reasoning": prefix = "reasoning"
    return f"{prefix}_{idx:03d}"

# ANIMALS (target 40-50 total, current 10)
# Let's generate 40 more
animals_data = [
    # domestic
    ("domestic", "atomic", "easy", "horse"),
    ("domestic", "atomic", "easy", "rabbit"),
    ("domestic", "atomic", "easy", "hamster"),
    ("domestic", "atomic", "easy", "guinea pig"),
    ("domestic", "atomic", "easy", "cow"),
    ("domestic", "atomic", "easy", "sheep"),
    ("domestic", "atomic", "easy", "pig"),
    ("domestic", "atomic", "easy", "goat"),
    ("domestic", "contextual", "easy", "The horse gallops across the field."),
    ("domestic", "contextual", "easy", "The rabbit hops in the garden."),
    
    # wild
    ("wild", "atomic", "easy", "tiger"),
    ("wild", "atomic", "easy", "bear"),
    ("wild", "atomic", "easy", "wolf"),
    ("wild", "atomic", "easy", "fox"),
    ("wild", "atomic", "easy", "elephant"),
    ("wild", "atomic", "easy", "giraffe"),
    ("wild", "atomic", "easy", "zebra"),
    ("wild", "atomic", "easy", "rhino"),
    ("wild", "atomic", "easy", "hippo"),
    ("wild", "atomic", "easy", "leopard"),
    ("wild", "contextual", "easy", "The wolf howls at the moon."),
    ("wild", "contextual", "easy", "The elephant walks through the savanna."),
    
    # marine
    ("marine", "atomic", "easy", "whale"),
    ("marine", "atomic", "easy", "shark"),
    ("marine", "atomic", "easy", "octopus"),
    ("marine", "atomic", "easy", "squid"),
    ("marine", "atomic", "easy", "seal"),
    ("marine", "atomic", "easy", "walrus"),
    ("marine", "contextual", "easy", "The whale breaches the ocean surface."),
    ("marine", "contextual", "easy", "The shark swims quickly through the water."),
    
    # birds
    ("bird", "atomic", "easy", "hawk"),
    ("bird", "atomic", "easy", "owl"),
    ("bird", "atomic", "easy", "penguin"),
    ("bird", "atomic", "easy", "ostrich"),
    ("bird", "atomic", "easy", "parrot"),
    ("bird", "atomic", "easy", "pigeon"),
    ("bird", "atomic", "easy", "seagull"),
    ("bird", "atomic", "easy", "duck"),
    ("bird", "contextual", "easy", "The owl hoots in the dark forest."),
    ("bird", "contextual", "easy", "The penguin waddles on the ice.")
]

for i, (subcat, ptype, diff, prompt) in enumerate(animals_data):
    new_prompts.append({
        "id": generate_id("animals", i + start_id),
        "group": "animals",
        "subcategory": subcat,
        "prompt_type": ptype,
        "difficulty": diff,
        "prompt": prompt
    })

# OBJECTS (target 40-50, current group is 'vehicles' 10, we'll map objects to 'objects' group and rename 'vehicles' to 'objects' or keep it as objects)
# The user said: Objects (vehicles, tools, household objects, electronics)
objects_data = [
    # vehicles
    ("vehicles", "atomic", "easy", "truck"),
    ("vehicles", "atomic", "easy", "bus"),
    ("vehicles", "atomic", "easy", "bicycle"),
    ("vehicles", "atomic", "easy", "motorcycle"),
    ("vehicles", "atomic", "easy", "helicopter"),
    ("vehicles", "atomic", "easy", "submarine"),
    ("vehicles", "atomic", "easy", "tractor"),
    ("vehicles", "atomic", "easy", "scooter"),
    ("vehicles", "contextual", "easy", "The truck carries a heavy load."),
    ("vehicles", "contextual", "easy", "The bus stops to pick up passengers."),
    
    # tools
    ("tools", "atomic", "easy", "hammer"),
    ("tools", "atomic", "easy", "screwdriver"),
    ("tools", "atomic", "easy", "wrench"),
    ("tools", "atomic", "easy", "pliers"),
    ("tools", "atomic", "easy", "saw"),
    ("tools", "atomic", "easy", "drill"),
    ("tools", "atomic", "easy", "chisel"),
    ("tools", "atomic", "easy", "level"),
    ("tools", "contextual", "easy", "He uses a hammer to drive the nail."),
    ("tools", "contextual", "easy", "The saw is used to cut the wood."),
    
    # household
    ("household", "atomic", "easy", "chair"),
    ("household", "atomic", "easy", "table"),
    ("household", "atomic", "easy", "sofa"),
    ("household", "atomic", "easy", "bed"),
    ("household", "atomic", "easy", "lamp"),
    ("household", "atomic", "easy", "plate"),
    ("household", "atomic", "easy", "fork"),
    ("household", "atomic", "easy", "spoon"),
    ("household", "contextual", "easy", "She sits on the comfortable sofa."),
    ("household", "contextual", "easy", "The dinner is served on a round plate."),
    
    # electronics
    ("electronics", "atomic", "easy", "computer"),
    ("electronics", "atomic", "easy", "laptop"),
    ("electronics", "atomic", "easy", "smartphone"),
    ("electronics", "atomic", "easy", "television"),
    ("electronics", "atomic", "easy", "radio"),
    ("electronics", "atomic", "easy", "camera"),
    ("electronics", "atomic", "easy", "microwave"),
    ("electronics", "atomic", "easy", "refrigerator"),
    ("electronics", "contextual", "easy", "He types on his laptop keyboard."),
    ("electronics", "contextual", "easy", "They watch a movie on the television.")
]

for i, (subcat, ptype, diff, prompt) in enumerate(objects_data):
    new_prompts.append({
        "id": generate_id("objects", i + start_id),
        "group": "objects",
        "subcategory": subcat,
        "prompt_type": ptype,
        "difficulty": diff,
        "prompt": prompt
    })

# REASONING (target 40-50, current reasoning is 10)
reasoning_data = [
    # transitive
    ("transitive", "reasoning", "medium", "If Alice is taller than Bob, and Bob is taller than Charlie, then Alice is taller than Charlie."),
    ("transitive", "reasoning", "medium", "If X is greater than Y, and Y is greater than Z, then X > Z."),
    ("transitive", "reasoning", "medium", "Since A is faster than B, and B is faster than C, it follows that A is faster than C."),
    ("transitive", "reasoning", "medium", "If Monday comes before Tuesday, and Tuesday comes before Wednesday, then Monday is before Wednesday."),
    ("transitive", "reasoning", "medium", "If David is older than Emma, and Emma is older than Frank, then David must be older than Frank."),
    ("transitive", "reasoning", "medium", "If Box 1 is heavier than Box 2, and Box 2 is heavier than Box 3, Box 1 is the heaviest."),
    ("transitive", "reasoning", "hard", "A implies B. B implies C. Therefore A implies C."),
    ("transitive", "reasoning", "hard", "If the red car is faster than the blue car, and the blue car beats the green car, the red car beats the green car."),
    
    # analogy
    ("analogy", "reasoning", "medium", "Cat is to kitten as dog is to puppy."),
    ("analogy", "reasoning", "medium", "Hot is to cold as high is to low."),
    ("analogy", "reasoning", "medium", "Sun is to day as moon is to night."),
    ("analogy", "reasoning", "medium", "Bird is to flying as fish is to swimming."),
    ("analogy", "reasoning", "medium", "Eye is to seeing as ear is to hearing."),
    ("analogy", "reasoning", "medium", "Doctor is to hospital as teacher is to school."),
    ("analogy", "reasoning", "hard", "Author is to book as composer is to symphony."),
    ("analogy", "reasoning", "hard", "Telescope is to astronomy as microscope is to biology."),
    
    # comparative
    ("comparative", "reasoning", "medium", "A cheetah runs faster than a lion, but a gazelle is faster than a zebra."),
    ("comparative", "reasoning", "medium", "Gold is more expensive than silver, but platinum is the most expensive."),
    ("comparative", "reasoning", "medium", "Summer is hotter than spring, while winter is the coldest season."),
    ("comparative", "reasoning", "medium", "A mountain is taller than a hill, but a skyscraper is taller than a house."),
    ("comparative", "reasoning", "medium", "Iron is heavier than aluminum, but lead is heavier than iron."),
    ("comparative", "reasoning", "medium", "Water is denser than oil, so oil floats on water."),
    ("comparative", "reasoning", "hard", "Jupiter is much larger than Earth, whereas Mercury is smaller than Earth."),
    ("comparative", "reasoning", "hard", "A gigabyte holds more data than a megabyte, but less than a terabyte."),
    
    # causal
    ("causal", "reasoning", "medium", "Because it rained heavily, the river flooded its banks."),
    ("causal", "reasoning", "medium", "If you heat ice to room temperature, it will melt into water."),
    ("causal", "reasoning", "medium", "The plant died because it did not receive any sunlight."),
    ("causal", "reasoning", "medium", "Dropping the glass on the hard floor caused it to shatter."),
    ("causal", "reasoning", "medium", "Eating too much candy will lead to a stomach ache."),
    ("causal", "reasoning", "medium", "Friction between the two surfaces generated a lot of heat."),
    ("causal", "reasoning", "hard", "The lack of regular maintenance resulted in the engine failing unexpectedly."),
    ("causal", "reasoning", "hard", "An increase in interest rates typically causes a decrease in consumer borrowing."),
    
    # arithmetic
    ("arithmetic", "reasoning", "easy", "If you have 5 apples and buy 3 more, you have 8 apples in total."),
    ("arithmetic", "reasoning", "easy", "10 minus 4 equals 6."),
    ("arithmetic", "reasoning", "easy", "Multiplying 7 by 3 gives 21."),
    ("arithmetic", "reasoning", "easy", "Dividing 20 by 4 results in 5."),
    ("arithmetic", "reasoning", "medium", "If a shirt costs $20 and is on a 50% discount, the new price is $10."),
    ("arithmetic", "reasoning", "medium", "A train travels 60 miles in one hour, so it goes 120 miles in two hours."),
    ("arithmetic", "reasoning", "hard", "If 3 workers can build a wall in 4 days, 6 workers can build it in 2 days."),
    ("arithmetic", "reasoning", "hard", "The sum of the first 5 positive integers is 15.")
]

for i, (subcat, ptype, diff, prompt) in enumerate(reasoning_data):
    new_prompts.append({
        "id": generate_id("reasoning", i + start_id),
        "group": "reasoning",
        "subcategory": subcat,
        "prompt_type": ptype,
        "difficulty": diff,
        "prompt": prompt
    })

# Read existing, filter and map
existing_prompts = []
with open('data/prompts/prompts.jsonl', 'r') as f:
    for line in f:
        p = json.loads(line)
        # Rename 'vehicles' to 'objects' group for existing vehicles to maintain 1 category mapping?
        # The user requested 'Objects' category. Let's update group 'vehicles' to 'objects'
        if p['group'] == 'vehicles':
            p['group'] = 'objects'
            # id is vehicle_xxx, keep or change? Let's just keep the id as is.
        existing_prompts.append(p)

all_prompts = existing_prompts + new_prompts

with open('data/prompts/prompts.jsonl', 'w') as f:
    for p in all_prompts:
        f.write(json.dumps(p) + '\n')

print(f"Total prompts now: {len(all_prompts)}")
counts = {}
for p in all_prompts:
    counts[p['group']] = counts.get(p['group'], 0) + 1
print("Counts per group:", counts)