import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
OUTPUT = PROJECT / "outputs" / "final"

categories = json.loads(
    (DATA / "public_categories.json").read_text(encoding="utf-8")
)
train_raw = pd.read_csv(DATA / "public_train.csv").dropna(
    subset=["text", "label"]
)
test_raw = pd.read_csv(DATA / "public_test.csv").dropna(
    subset=["text", "label"]
)

for frame in (train_raw, test_raw):
    frame["text"] = frame["text"].str.strip()
    frame["label"] = frame["label"].astype(int)

# Exclude test text that appeared anywhere in the original training set.
test = test_raw.loc[
    ~test_raw["text"].isin(set(train_raw["text"]))
].copy()

# Remove ambiguous labels and repeated texts from training.
conflicts = train_raw.groupby("text")["label"].nunique()
conflicting_texts = set(conflicts[conflicts > 1].index)
train = train_raw.loc[
    ~train_raw["text"].isin(conflicting_texts)
].drop_duplicates(subset=["text"]).copy()

# EOL has only two distinct templates and no unseen test examples.
eol_id = categories.index("EOL")
train = train.loc[train["label"] != eol_id].copy()

# Count repeated and conflicting texts in the remaining test set.
test_conflicts = test.groupby("text")["label"].nunique()
ambiguous_test_texts = set(test_conflicts[test_conflicts > 1].index)
test = test.loc[
    ~test["text"].isin(ambiguous_test_texts)
].drop_duplicates(subset=["text"]).copy()

labels = sorted(train["label"].unique())
names = [categories[i] for i in labels]

model = make_pipeline(
    TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=2,
    ),
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    ),
)
model.fit(train["text"], train["label"])
predicted = model.predict(test["text"])

report = classification_report(
    test["label"], predicted,
    labels=labels,
    target_names=names,
    zero_division=0,
    output_dict=True,
)
matrix = confusion_matrix(test["label"], predicted, labels=labels)

print(f"Final training tickets: {len(train)}")
print(f"Original test tickets: {len(test_raw)}")
print(f"Unseen, unique, unambiguous test tickets: {len(test)}")
print(f"Excluded ambiguous training texts: {len(conflicting_texts)}")
print(f"Excluded ambiguous test texts: {len(ambiguous_test_texts)}")
print("\nFINAL CHARACTER MODEL")
print(classification_report(
    test["label"], predicted,
    labels=labels,
    target_names=names,
    zero_division=0,
))
print("Confusion matrix category order:", names)
print(matrix)

OUTPUT.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {"model": model, "categories": categories, "supported_labels": labels},
    OUTPUT / "model.joblib",
)

errors = pd.DataFrame({
    "text": test["text"].to_numpy(),
    "actual": [categories[i] for i in test["label"]],
    "predicted": [categories[i] for i in predicted],
})
errors.loc[errors["actual"] != errors["predicted"]].to_csv(
    OUTPUT / "errors.csv", index=False
)

(OUTPUT / "metrics.json").write_text(json.dumps({
    "dataset": "tasksource/it-support-tickets",
    "model": "character TF-IDF (3–5) + balanced logistic regression",
    "train_count": len(train),
    "original_test_count": len(test_raw),
    "evaluated_test_count": len(test),
    "categories": names,
    "report": report,
    "confusion_matrix": matrix.tolist(),
    "limitation": "EOL excluded: two unique training templates and no unseen test text",
}, indent=2), encoding="utf-8")

print(f"\nSaved model, metrics, and errors in {OUTPUT}")