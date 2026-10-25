import csv
import json
import re
from functools import lru_cache
from pathlib import Path

from datasets import load_dataset

# 1. Locate files
project_root = Path(__file__).resolve().parents[1]
json_path = project_root / "data" / "model_profiles.json"
output_csv = project_root / "data" / "router_training_pairs_scrubbed.csv"

# 2. Load the real OpenRouter profiles
print("Loading OpenRouter profiles...")
with json_path.open("r", encoding="utf-8") as f:
    profiles = json.load(f)

@lru_cache(maxsize=None)
def get_aggressively_scrubbed_description(arena_model_name):
    best_match_profile = None
    
    # Find matching OpenRouter profile based on the Arena ID
    for or_id, profile in profiles.items():
        short_id = or_id.split("/")[-1].lower()
        if short_id in arena_model_name.lower() or arena_model_name.lower() in short_id:
            best_match_profile = profile
            break
            
    if not best_match_profile or not best_match_profile.get("description"):
        return None
        
    desc = best_match_profile["description"].strip()
    
    # Reject incomplete API descriptions ending in ellipses
    if desc.endswith("..."):
        return None
    
    # Greedy Regex: Swallows attached numbers, hyphens, and dots (e.g., Qwen3-30B)
    model_roots = [
        "gpt", "claude", "llama", "qwen", "mistral", "mixtral", 
        "gemma", "gemini", "phi", "grok", "yi", "command", "nemotron"
    ]
    roots_pattern = r'(?i)\b(?:' + '|'.join(model_roots) + r')[a-zA-Z0-9\-\.]*'
    desc = re.sub(roots_pattern, "this AI model", desc)
    
    # Scrub provider companies
    providers = [
        "OpenAI", "Anthropic", "Google", "Meta", "Mistral", 
        "Cohere", "Qwen", "Microsoft", "Alibaba", "xAI", "Tencent"
    ]
    provider_pattern = r'(?i)\b(?:' + '|'.join(providers) + r')\b'
    desc = re.sub(provider_pattern, "the developer", desc)
    
    # Scrub exact name strings that might have slipped through
    name = best_match_profile.get("name", "")
    if len(name) > 3:
        desc = re.sub(re.escape(name), "this AI model", desc, flags=re.IGNORECASE)
        
    return desc

# 4. Load the Hugging Face arena dataset
print("Downloading/Loading 140k dataset from Hugging Face...")
dataset = load_dataset("lmarena-ai/arena-human-preference-140k", split="train")

# 5. Process and generate pairs
print("Processing pairs and aggressively scrubbing names...")
training_data = []
max_pairs = 15000

for row in dataset:
    if len(training_data) + 2 > max_pairs:
        break
        
    try:
        prompt = row["conversation_a"][0]["content"][0]["text"]
    except (IndexError, KeyError, TypeError):
        continue
        
    winner = row["winner"]
    if winner not in ["model_a", "model_b"]:
        continue

    print(f"Processing battle: {row['model_a']} vs {row['model_b']} | Winner: {winner}, battle: len={len(training_data)}")
    desc_a = get_aggressively_scrubbed_description(row["model_a"])
    desc_b = get_aggressively_scrubbed_description(row["model_b"])
    
    # Append if both descriptions exist and do not end in ellipses
    if desc_a and desc_b:
        if winner == "model_a":
            training_data.append([prompt, desc_a, 1.0])
            training_data.append([prompt, desc_b, 0.0])
        else:
            training_data.append([prompt, desc_b, 1.0])
            training_data.append([prompt, desc_a, 0.0])

        if len(training_data) >= max_pairs:
            break

# 6. Save the new dataset
with output_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["sentence1", "sentence2", "label"])
    writer.writerows(training_data)

print(f"Success! Generated {len(training_data)} thoroughly scrubbed training pairs.")
print(f"Saved to: {output_csv}")