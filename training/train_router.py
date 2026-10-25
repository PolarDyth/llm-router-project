import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder

# 1. Load the dataset
df = pd.read_csv("C:\\Users\\Internet Cafe\\Documents\\projects\\llm-router-project\\data\\router_training_pairs.csv")

# 2. Convert the Pandas rows into native InputExamples
train_examples = []
for _, row in df.iterrows():
    train_examples.append(
        InputExample(texts=[str(row["sentence1"]), str(row["sentence2"])], label=float(row["label"]))
    )

# 3. Create a PyTorch DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

# 4. Initialize the base model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)

# 5. Run the native training loop
print("Starting training...")
model.fit(
    train_dataloader=train_dataloader,
    epochs=3,
    optimizer_params={'lr': 2e-5},
    warmup_steps=100
    # Removed output_path from here since it relies on an evaluator
)

# 6. Explicitly save the model at the end
model.save("./models/dynamic_router/final")
print("Training complete! Model saved.")