import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

# Paths work regardless of where VS Code launches the script.
PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data"
OUTPUT_DIR = PROJECT / "outputs" / "real"
TRAIN_FILE = DATA_DIR / "public_train.csv"
TEST_FILE = DATA_DIR / "public_test.csv"
CATEGORIES_FILE = DATA_DIR / "public_categories.json"


def load_local_data():
    if not all(path.exists() for path in (TRAIN_FILE, TEST_FILE, CATEGORIES_FILE)):
        print("Downloading dataset once and saving local copies...")
        from datasets import load_dataset

        dataset = load_dataset("tasksource/it-support-tickets")
        categories = dataset["train"].features["label"].names

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dataset["train"].to_pandas()[["text", "label"]].to_csv(TRAIN_FILE, index=False)
        dataset["test"].to_pandas()[["text", "label"]].to_csv(TEST_FILE, index=False)
        CATEGORIES_FILE.write_text(
            json.dumps(categories, indent=2), encoding="utf-8"
        )
    else:
        print("Using locally saved dataset.")

    train = pd.read_csv(TRAIN_FILE).dropna(subset=["text", "label"])
    test = pd.read_csv(TEST_FILE).dropna(subset=["text", "label"])
    categories = json.loads(CATEGORIES_FILE.read_text(encoding="utf-8"))

    train["label"] = train["label"].astype(int)
    test["label"] = test["label"].astype(int)
    return train, test, categories


def main():
    train, original_test, categories = load_local_data()

    # Remove test examples whose text appears in training.
    train_text_set = set(train["text"].str.strip())
    overlap = original_test["text"].str.strip().isin(train_text_set)
    test = original_test.loc[~overlap].copy()

    print(f"\nOriginal split: {len(train)} train / {len(original_test)} test")
    print(f"Exact train–test text matches removed: {overlap.sum()}")
    print(f"Clean test tickets: {len(test)}")

    eol_id = categories.index("EOL")
    train_eol = train.loc[train["label"] == eol_id, "text"]
    test_eol = original_test.loc[original_test["label"] == eol_id, "text"]
    print(
        f"\nEOL: {len(train_eol)} train tickets "
        f"({train_eol.nunique()} unique texts); "
        f"{len(test_eol)} original test tickets "
        f"({test_eol.nunique()} unique texts)"
    )
    if not train_eol.empty:
        print("Example EOL ticket:", repr(train_eol.iloc[0][:250]))

    train_text = train["text"].tolist()
    train_labels = train["label"].tolist()
    test_text = test["text"].tolist()
    test_labels = test["label"].tolist()

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit([[0]] * len(train_labels), train_labels)
    dummy_predictions = dummy.predict([[0]] * len(test_labels))

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(train_text, train_labels)
    predictions = model.predict(test_text)

    # Only score categories that have examples in the clean test set.
    evaluated_labels = sorted(set(test_labels))
    evaluated_names = [categories[i] for i in evaluated_labels]

    dummy_report = classification_report(
        test_labels,
        dummy_predictions,
        labels=evaluated_labels,
        target_names=evaluated_names,
        zero_division=0,
        output_dict=True,
    )
    model_report = classification_report(
        test_labels,
        predictions,
        labels=evaluated_labels,
        target_names=evaluated_names,
        zero_division=0,
        output_dict=True,
    )

    print("\nDUMMY BASELINE")
    print(classification_report(
        test_labels, dummy_predictions,
        labels=evaluated_labels,
        target_names=evaluated_names,
        zero_division=0,
    ))

    print("\nTF-IDF + LOGISTIC REGRESSION")
    print(classification_report(
        test_labels, predictions,
        labels=evaluated_labels,
        target_names=evaluated_names,
        zero_division=0,
    ))

    matrix = confusion_matrix(
        test_labels, predictions, labels=evaluated_labels
    )
    print("Confusion matrix — rows actual, columns predicted:")
    print("Category order:", evaluated_names)
    print(matrix)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "categories": categories},
        OUTPUT_DIR / "model.joblib",
    )

    errors = pd.DataFrame({
        "text": test_text,
        "actual": [categories[i] for i in test_labels],
        "predicted": [categories[i] for i in predictions],
    })
    errors = errors.loc[errors["actual"] != errors["predicted"]]
    errors.to_csv(OUTPUT_DIR / "errors.csv", index=False)

    metrics = {
        "train_count": len(train),
        "original_test_count": len(original_test),
        "overlapping_test_count": int(overlap.sum()),
        "clean_test_count": len(test),
        "evaluated_categories": evaluated_names,
        "dummy": dummy_report,
        "model": model_report,
        "confusion_matrix": matrix.tolist(),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"\nSaved model, metrics and {len(errors)} errors to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()