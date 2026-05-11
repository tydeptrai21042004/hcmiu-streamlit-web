# ONNX deployment notes

This version supports `calf_weather_forecast.onnx` inference through `onnxruntime`, so the Streamlit deployment does not need `torch`, `transformers`, or `peft`.

## How to use

1. Export the Weather CALF model to ONNX in Colab.
2. Put the ONNX file in one of these places:
   - `weights/calf_weather_forecast.onnx` inside this app, or
   - upload/import it through the **Import required files** tab.
3. Open the **ONNX inference** tab.
4. Run **Run ONNX Weather inference**.

## Recognized ONNX names

The importer recognizes these names:

- `calf_weather_forecast.onnx`
- `calf_forecast.onnx`
- `model.onnx`
- any `.onnx` file
- any `.onxr` file, saved internally as `.onnx`

All are saved to:

```text
weights/calf_weather_forecast.onnx
```

## Minimal dependencies

```text
streamlit
numpy
pandas
gdown
onnxruntime
```

No PyTorch is required for the ONNX inference tab.
