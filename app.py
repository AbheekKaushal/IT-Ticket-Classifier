from pathlib import Path

import joblib
import streamlit as st

MODEL_FILE = Path(__file__).resolve().parent / "outputs" / "final" / "model.joblib"

st.set_page_config(page_title="IT Ticket Classifier", page_icon="🎫")
st.title("AI IT Support Ticket Classifier")
st.caption("Character TF-IDF + logistic regression · six supported categories")

if not MODEL_FILE.exists():
    st.error("Run ticket_classifier/final_train.py before opening the app.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

saved = load_model()
model = saved["model"]
categories = saved["categories"]

ticket = st.text_area(
    "Describe the IT issue",
    height=150,
    placeholder="Example: My account is locked and I cannot sign in.",
)

if st.button("Classify ticket", type="primary"):
    if not ticket.strip():
        st.warning("Enter a ticket description first.")
    else:
        label_id = int(model.predict([ticket])[0])
        scores = model.predict_proba([ticket])[0]
        ranked = sorted(
            zip(model.classes_, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        top_score = ranked[0][1]

        if top_score < 0.40:
            st.warning(
                "Needs human review — this ticket may not fit the supported categories."
            )
        else:
            st.success(f"Suggested category: **{categories[label_id]}**")
        st.write("Top suggestions:")
        for category_id, score in ranked[:3]:
            st.write(f"- {categories[int(category_id)]}: {score:.1%}")

        st.caption(
            "These are model scores, not verified confidence. "
            "A support agent should review uncertain or unusual tickets. "
            "EOL is outside this model's supported categories."
        )