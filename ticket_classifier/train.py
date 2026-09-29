import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def train(data: Path, output: Path) -> dict:
    frame = pd.read_csv(data)
    if not {'text', 'category'}.issubset(frame.columns):
        raise ValueError('CSV requires text and category columns')
    frame = frame[['text', 'category']].dropna().copy()
    frame['text'] = frame['text'].astype(str).str.strip()
    frame['category'] = frame['category'].astype(str).str.strip()
    frame = frame[(frame.text != '') & (frame.category != '')]
    conflicts = frame.groupby('text').category.nunique()
    if (conflicts > 1).any():
        raise ValueError('Identical ticket text has conflicting category labels')
    frame = frame.drop_duplicates(subset=['text'])
    counts = frame.category.value_counts()
    if len(counts) < 2 or counts.min() < 5:
        raise ValueError('Need at least two categories and five distinct tickets per category')
    if len(frame) * 0.25 < len(counts):
        raise ValueError('Need enough examples to include every category in the test split')

    x_train, x_test, y_train, y_test = train_test_split(
        frame.text, frame.category, test_size=0.25, stratify=frame.category, random_state=42
    )
    baseline = DummyClassifier(strategy='most_frequent').fit(x_train.to_frame(), y_train)
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    ).fit(x_train, y_train)
    labels = sorted(counts.index.tolist())
    predictions = model.predict(x_test)
    metrics = {
        'note': 'Demo scores are not a real-world performance estimate' if 'demo' in data.name else 'Single held-out split; validate on independent data before deployment',
        'samples_after_cleaning': len(frame), 'class_counts': counts.to_dict(),
        'train_size': len(x_train), 'test_size': len(x_test), 'labels': labels,
        'dummy': classification_report(y_test, baseline.predict(x_test.to_frame()), labels=labels, output_dict=True, zero_division=0),
        'tfidf_logistic': classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, predictions, labels=labels).tolist(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({'text': x_test.values, 'actual': y_test.values, 'predicted': predictions}).query('actual != predicted').to_csv(output / 'errors.csv', index=False)
    joblib.dump(model, output / 'model.joblib')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = train(args.data, args.output)
    print(json.dumps({'samples': result['samples_after_cleaning'], 'macro_f1': result['tfidf_logistic']['macro avg']['f1-score'], 'dummy_macro_f1': result['dummy']['macro avg']['f1-score']}, indent=2))
