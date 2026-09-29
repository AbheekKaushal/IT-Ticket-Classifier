# IT Support Ticket Classifier

A Streamlit app that suggests a category for an incoming IT support ticket. It uses character-level TF-IDF (3–5 grams) and balanced logistic regression trained on the public `tasksource/it-support-tickets` dataset.

## Results

| Metric | Held-out test result |
| --- | ---: |
| Accuracy | **77%** |
| Macro precision | **0.75** |
| Macro recall | **0.75** |
| Macro F1 | **0.74** |
| Weighted precision | **0.78** |
| Weighted F1 | **0.77** |

The evaluation used **522 unseen, unique tickets** after removing train–test text overlap, duplicate test text, and ambiguous labels. The final model was trained on **953 unique tickets**. These scores measure agreement with this dataset's labels, not performance on every organisation's help desk tickets.

| Category | Precision | Recall | F1 | Test tickets |
| --- | ---: | ---: | ---: | ---: |
| Active Directory | 0.54 | 0.51 | 0.52 | 43 |
| Computer-Services | 0.89 | 0.83 | 0.86 | 41 |
| Fileservice | 0.95 | 0.89 | 0.92 | 114 |
| O365 | 0.66 | 0.82 | 0.73 | 89 |
| Software | 0.62 | 0.67 | 0.64 | 58 |
| Support general | 0.82 | 0.76 | 0.79 | 177 |

`EOL` was excluded: its training examples contain only two distinct templates and no unseen test example remained. The model has no `Network` category, so a VPN issue cannot be classified as Network. The app flags low-scoring suggestions for review; its 0.40 cutoff is a display heuristic, not a calibrated confidence threshold.

## Run the app

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

The included trained model lets you try the app immediately. Only load model files from sources you trust (`joblib` deserialization can execute code).

## Reproduce the evaluation

`train_public.py` downloads the public dataset on the first run and stores local CSV files in `data/`. Validation compares word and character n-grams using training data only. The character model scored **0.692 macro F1** on validation versus **0.618** for the word model.

```bash
python ticket_classifier/train_public.py
python ticket_classifier/validate_models.py
python ticket_classifier/final_train.py
```

The final script writes the model and full metrics to `outputs/final/`. The local dataset and ticket-level error files are excluded from Git. Dataset labels reflect the source organisation's routing rules and may not match another help desk's taxonomy.
