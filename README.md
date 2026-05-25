# CALF Weather Forecast — Clean Streamlit UI

A simplified Streamlit app for CALF Weather forecasting.

## Interface

The app uses a clean 3-step workflow:

1. **Upload** data/model/result files.
2. **Forecast** with the ONNX model, or run Demo mode only to test the UI.
3. **Results & data** to view charts, metrics, tables, and download CSV output.

## Supported files

```text
weather.csv                  Weather input data
calf_weather_forecast.onnx   ONNX model for real CALF inference
pred.npy                     Saved prediction array
true.npy                     Saved ground-truth array
input.npy                    Optional saved input/history array
metrics.npy                  Optional saved global metrics
.zip                         Archive containing any files above
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

Deploy this folder and set the entry point to:

```text
app.py
```

The app is designed for ONNX deployment, so it does not require `torch`, `transformers`, or `peft`.

## Important note

Demo forecast is only for checking the user interface. For real CALF forecasting, upload/export:

```text
weights/calf_weather_forecast.onnx
```
