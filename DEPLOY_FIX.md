# Streamlit Cloud deployment fix

This is a lightweight Cloud version. It removes `matplotlib`, `scikit-learn`, `torch`, `transformers`, and `peft` from `requirements.txt` so the app starts quickly.

Use this version to:

- test the Streamlit UI,
- upload/view Weather CSV,
- run the placeholder forecast,
- load saved `pred.npy`, `true.npy`, `input.npy`, and `metrics.npy` results.

For real CALF inference, run the CALF model locally, in Colab, or in Kaggle, then upload/load the saved result arrays in Streamlit Cloud.

Recommended Streamlit Cloud setting:

- Python version: 3.12
- Main file: `app.py`

After changing the Python version, redeploy the app. If Streamlit Cloud does not allow changing Python version for an existing app, delete the app and redeploy with Python 3.12 selected in Advanced settings.
