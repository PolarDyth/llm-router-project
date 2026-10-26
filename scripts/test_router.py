import json
import sys
from pathlib import Path

from sentence_transformers.cross_encoder import CrossEncoder

# 1. Dynamically locate the project folders
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from device_utils import select_device

model_path = project_root / "models" / "dynamic_router" / "final"
json_path = project_root / "data" / "model_profiles.json"

# 2. Load the trained model and the JSON database
device = select_device()
print("Loading dynamic router model...")
# sentence-transformers accepts torch.device at runtime; the __init__
# annotation is narrower than reality.
model = CrossEncoder(str(model_path), device=device)  # pyright: ignore[reportArgumentType]

print("Loading scraped model profiles...")
with json_path.open("r", encoding="utf-8") as f:
    model_profiles: dict[str, dict[str, str]] = json.load(f)

# 3. Define the test prompt
user_prompt = "you are a helpful assistant that can answer questions about LLMs. I want to know which model is best for answering questions about the latest research in AI and machine learning."

# 4. Build pairs for every model that has a description
model_names: list[str] = []
pairs: list[list[str]] = []
for name, data in model_profiles.items():
    desc = data.get("description", "")
    # Skip entries with missing descriptions
    if desc.strip():
        model_names.append(name)
        pairs.append([user_prompt, desc])

# 5. Generate scores for the entire database
print(f"Calculating scores against {len(pairs)} candidate models...")
# predict() already applies the model's sigmoid activation (num_labels=1),
# so these are probabilities in [0, 1] — no extra sigmoid needed.
scores: list[float] = [float(s) for s in model.predict(pairs)]

# 6. Rank and display the top 5 results
results = sorted(zip(model_names, scores), key=lambda item: item[1], reverse=True)

print(f"\nUser Prompt: '{user_prompt}'\n")
print("Top 20 Recommended Models:")
for rank, (name, score) in enumerate(results[:20], 1):
    desc = model_profiles[name].get("description", "No description")
    # Truncate the description if it's too long for the terminal
    short_desc = (desc[:100] + '...') if len(desc) > 100 else desc
    print(f"{rank}. {name}: {score * 100:.2f}%")
    print(f"   Context: {short_desc}\n")