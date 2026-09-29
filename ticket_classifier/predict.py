import argparse
from pathlib import Path
import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--text', required=True)
    args = parser.parse_args()
    model = joblib.load(args.model)
    probabilities = model.predict_proba([args.text])[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda item: item[1], reverse=True)
    print(f'Predicted category: {ranked[0][0]}')
    print('Model probabilities: ' + ', '.join(f'{label} {score:.1%}' for label, score in ranked))
