import json
import csv
import random
from pathlib import Path

# 1. Locate files
project_root = Path(__file__).resolve().parents[1]
json_path = project_root / "data" / "model_profiles.json"
output_csv = project_root / "data" / "router_training_pairs_v2.csv"

# 2. Load the real OpenRouter profiles
with json_path.open("r", encoding="utf-8") as f:
    profiles = json.load(f)

# Extract real descriptions, ignoring empty ones
valid_models = [data["description"] for data in profiles.values() if data.get("description")]

# 3. Define prompt categories to teach the router
prompt_types = {
    "coding": [
        "Write a complex PyTorch script to train a neural network with distributed data parallel.",
        "Can you debug this React useEffect hook that causes an infinite loop?",
        "Write a Python script to scrape a website using BeautifulSoup.",
        "How do I implement a binary search tree in C++?"
    ],
    "casual": [
        "opowiedz mi żart o gołębiach",
        "What is a good recipe for chocolate chip cookies?",
        "Write a polite email to my boss asking for next Friday off.",
        "mowisz po polsku?"
    ],
    "reasoning": [
        "Factory A can produce 45 machines in 20 minutes, while factory B can produce 30 machines in 15 minutes. If both factories operate for 2 hours, which factory will produce more machines?",
        "ein zusammenhängender graph kann keine isolierten knoten haben, ist dieser beweis dennoch korrekt?",
        "What are the philosophical implications of the Ship of Theseus?",
        "Explain the cascading analysis of security events."
    ]
}

# 4. Generate intelligent training pairs
training_data = []

for category, prompts in prompt_types.items():
    for prompt in prompts:
        # Sample 10 random models for each prompt to evaluate
        sampled_models = random.sample(valid_models, min(10, len(valid_models)))
        
        for desc in sampled_models:
            desc_lower = desc.lower()
            
            # Label = 1 if the description contains keywords matching the prompt category
            if category == "coding" and any(k in desc_lower for k in ["code", "coder", "programming", "developer", "python"]):
                label = 1.0
            elif category == "reasoning" and any(k in desc_lower for k in ["reasoning", "logic", "advanced", "flagship", "math"]):
                label = 1.0
            elif category == "casual" and any(k in desc_lower for k in ["fast", "lightweight", "chat", "general", "everyday"]):
                label = 1.0
            else:
                label = 0.0
                
            training_data.append([prompt, desc, label])

# 5. Save the new dataset
with output_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["sentence1", "sentence2", "label"])
    writer.writerows(training_data)

print(f"Success! Generated {len(training_data)} highly-targeted training pairs.")
print(f"Saved to: {output_csv}")