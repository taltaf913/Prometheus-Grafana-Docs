import streamlit as st
from translation_model import TranslationModel
import time
from prometheus_client import start_http_server, Summary, Gauge
import logging
import evidently
from evidently.model_performance import DataDriftDetector
import pandas as pd
from bleu_score import calculate_bleu_score

# --- Prometheus Setup ---
start_http_server(8000)
REQUEST_TIME = Summary('translation_request_processing_seconds', 'Time spent processing translation request')
BLEU_SCORE = Gauge('translation_bleu_score', 'BLEU score of the translation')

# --- Evidently AI Setup ---
BASELINE_DATA = pd.read_csv("baseline_data.csv")
drift_detector = DataDriftDetector(columns=BASELINE_DATA.columns)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Initialization ---
model = TranslationModel()
REFERENCE_TRANSLATION = "This is a sample translation."  # Replace with a known good translation

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        REQUEST_TIME.observe(end - start)
        return result
    return wrapper

@monitor_performance
def translate_text(text):
    translation = model.translate(text)
    logging.info(f"Translated: {text} -> {translation}")

    # Calculate BLEU score
    bleu = calculate_bleu_score(REFERENCE_TRANSLATION, translation)
    BLEU_SCORE.set(bleu)  # Expose BLEU score as a Prometheus metric

    return translation

st.title("Translation App with Monitoring")

input_text = st.text_area("Enter text to translate:", "Hello, world!")

if st.button("Translate"):
    with REQUEST_TIME:
        translation = translate_text(input_text)
        st.write("Translation:", translation)

    # --- Data Drift Monitoring ---
    new_data = pd.DataFrame([input_text], columns=['text'])
    drift_results = drift_detector.drift(BASELINE_DATA, new_data)
    st.write("Data Drift Report:")
    st.write(drift_results)
