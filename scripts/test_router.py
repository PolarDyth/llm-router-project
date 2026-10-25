import json
import torch
from pathlib import Path
from sentence_transformers.cross_encoder import CrossEncoder

# 1. Dynamically locate the project folders
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
model_path = project_root / "models" / "dynamic_router" / "final"
json_path = project_root / "data" / "model_profiles.json"

# 2. Load the trained model and the JSON database
print("Loading dynamic router model...")
model = CrossEncoder(str(model_path))

print("Loading scraped model profiles...")
with json_path.open("r", encoding="utf-8") as f:
    model_profiles = json.load(f)

# 3. Define the test prompt
user_prompt = "Write a complex PyTorch script to train a neural network with distributed data parallel."

# 4. Build pairs for every model that has a description
model_names = []
pairs = []
for name, data in model_profiles.items():
    desc = data.get("description", "")
    # Skip entries with missing descriptions
    if desc.strip():
        model_names.append(name)
        pairs.append([user_prompt, desc])

# 5. Generate scores for the entire database
print(f"Calculating scores against {len(pairs)} candidate models...")
raw_scores = model.predict(pairs)
scores = torch.sigmoid(torch.tensor(raw_scores)).tolist()

# 6. Rank and display the top 5 results
results = list(zip(model_names, scores))
results.sort(key=lambda x: x[1], reverse=True)

print(f"\nUser Prompt: '{user_prompt}'\n")
print("Top 5 Recommended Models:")
for rank, (name, score) in enumerate(results[:5], 1):
    desc = model_profiles[name].get("description", "No description")
    # Truncate the description if it's too long for the terminal
    short_desc = (desc[:100] + '...') if len(desc) > 100 else desc
    print(f"{rank}. {name}: {score * 100:.2f}%")
    print(f"   Context: {short_desc}\n")