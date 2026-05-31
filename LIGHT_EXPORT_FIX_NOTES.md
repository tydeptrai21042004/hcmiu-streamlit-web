# Light Export Fix Notes

The old app had three main problems:

1. It used a 70% train split to refit the scaler, while the notebook uses 80/10/10.
2. It did not import `weather_scaler.npz` or `metadata.json` from the lightweight export.
3. It could mix new `pred.npy` with old `true.npy`/`input.npy` in the Streamlit session.

This corrected app fixes those issues by reading the scaler and metadata from the same export ZIP and clearing stale result files before importing a new pack.
