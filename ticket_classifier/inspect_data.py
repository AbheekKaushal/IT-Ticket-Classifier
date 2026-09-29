from collections import Counter
from datasets import load_dataset

data = load_dataset("tasksource/it-support-tickets")

print(data)
print("Categories:", data["train"].features["label"].names)

for split_name, split in data.items():
    names = split.features["label"].names
    counts = Counter(names[row["label"]] for row in split)
    blanks = sum(not row["text"].strip() for row in split)

    print(f"\n{split_name}: {len(split)} tickets")
    print("Category counts:", dict(sorted(counts.items())))
    print("Blank tickets:", blanks)

print("\nExample ticket:", data["train"][0]["text"][:300])