# Realistic Demo Forecast Fix

This version replaces the old flat demo forecast with a stronger non-ML seasonal baseline.

## What changed

The old demo forecast used only:

```python
last_value + local_slope * steps + small sine wave
```

That often looked almost horizontal and made the UI appear broken.

The new demo forecast uses:

1. inferred seasonal period from the observed history,
2. daily 24-step pattern,
3. weekly 168-step pattern when available,
4. robust local trend from median recent differences,
5. continuity correction so the forecast starts close to the last observed point,
6. clipping to a plausible recent range.

## Important note

This is still not CALF inference. It is a realistic fallback baseline for demo/UI purposes only.
For the real model demo, upload either:

- `CALF_weather_light_export.zip` and use **Step 3 - Saved results**, or
- a compatible `.onnx` model and use **Step 2 - ONNX model**.
