import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

PROJECT = Path(__file__).resolve().parents[1]
TRAIN_FILE = PROJECT / "data" / "public_train.csv"
CATEGORIES_FILE = PROJECT / "data" / "public_categories.json"
OUTPUT_FILE = PROJECT / "outputs" / "real" / "validation.json"

train = pd.read_csv(TRAIN_FILE).dropna(subset=["text", "label"])
categories = json.loads(CATEGORIES_FILE.read_text(encoding="utf-8"))
train["label"] = train["label"].astype(int)
train["text"] = train["text"].str.strip()

# Identical text with different labels cannot provide an unambiguous example.
conflicting = train.groupby("text")["label"].nunique()
conflicting_texts = set(conflicting[conflicting > 1].index)
train = train.loc[~train["text"].isin(conflicting_texts)].copy()

# Keep identical text out of both sides of the validation split.
train = train.drop_duplicates(subset=["text"])

eol_id = categories.index("EOL")
train = train.loc[train["label"] != eol_id].copy()

print(f"Conflicting texts removed: {len(conflicting_texts)}")
print(f"Unique non-EOL training tickets: {len(train)}")
print("Counts:", {
    categories[label]: count
    for label, count in train["label"].value_counts().sort_index().items()
})

x_fit, x_val, y_fit, y_val = train_test_split(
    train["text"],
    train["label"],
    test_size=0.20,
    stratify=train["label"],
    random_state=42,
)

models = {
    "word_ngrams": make_pipeline(
        TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            sublinear_tf=True, min_df=2
        ),
        LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
    ),
    "character_ngrams": make_pipeline(
        TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            sublinear_tf=True, min_df=2
        ),
        LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
    ),
}

results = {}
labels = sorted(train["label"].unique())
names = [categories[label] for label in labels]

for name, model in models.items():
    model.fit(x_fit, y_fit)
    predicted = model.predict(x_val)

    score = f1_score(y_val, predicted, labels=labels, average="macro")
    results[name] = {
        "macro_f1": score,
        "report": classification_report(
            y_val, predicted,
            labels=labels,
            target_names=names,
            zero_division=0,
            output_dict=True,
        ),
    }

    print(f"\n{name} — validation macro F1: {score:.3f}")
    print(classification_report(
        y_val, predicted,
        labels=labels,
        target_names=names,
        zero_division=0,
    ))

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text(json.dumps({
    "method": "20% stratified split of unique, non-EOL training texts",
    "fit_count": len(x_fit),
    "validation_count": len(x_val),
    "conflicting_texts_removed": len(conflicting_texts),
    "results": results,
}, indent=2), encoding="utf-8")

print(f"\nSaved validation results to {OUTPUT_FILE}")
print("The original test set was not used.")