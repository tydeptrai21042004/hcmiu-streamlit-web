"""Clean Streamlit UI for CALF Weather Forecasting.

This version is intentionally simple for non-technical users:
1. Upload a Weather CSV and/or ONNX model.
2. Run a forecast.
3. View and download results.

The old legacy PyTorch CALF runner was removed from the UI because it needs
heavy dependencies and many project files. Streamlit Cloud deployment should
use ONNX inference or saved result arrays.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="CALF Weather Forecast",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
WEIGHT_DIR = APP_DIR / "weights"
RESULT_DIR = APP_DIR / "results"

USER_CSV_PATH = DATA_DIR / "weather.csv"
SAMPLE_CSV_PATH = DATA_DIR / "weather_sample.csv"
LEGACY_CSV_PATH = APP_DIR / "CALF" / "datasets" / "weather" / "weather.csv"
ONNX_PATH = WEIGHT_DIR / "calf_weather_forecast.onnx"

METRIC_NAMES = ["MAE", "MSE", "RMSE", "MAPE", "MSPE"]
RECOGNIZED_RESULT_FILES = {"pred.npy", "true.npy", "input.npy", "metrics.npy"}


# ============================================================
# Lightweight styling
# ============================================================

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1180px;}
        [data-testid="stMetric"] {
            background: rgba(250, 250, 250, 0.65);
            border: 1px solid rgba(49, 51, 63, 0.10);
            padding: 0.85rem 1rem;
            border-radius: 0.85rem;
        }
        .small-note {
            color: rgba(49, 51, 63, 0.70);
            font-size: 0.92rem;
            line-height: 1.45;
        }
        .success-box {
            padding: 0.9rem 1rem;
            border-radius: 0.85rem;
            background: rgba(46, 204, 113, 0.10);
            border: 1px solid rgba(46, 204, 113, 0.25);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Data/scaling helpers
# ============================================================

@dataclass
class ForecastConfig:
    target: str
    seq_len: int
    pred_len: int
    backtest: bool
    inverse_scale_saved_arrays: bool


class SimpleStandardScaler:
    """Tiny StandardScaler replacement to keep deployment lightweight."""

    def fit(self, values: np.ndarray) -> "SimpleStandardScaler":
        arr = np.asarray(values, dtype=float)
        self.mean_ = np.nanmean(arr, axis=0)
        self.scale_ = np.nanstd(arr, axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale_ + self.mean_


def ensure_dirs() -> None:
    for path in [DATA_DIR, WEIGHT_DIR, RESULT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def create_synthetic_weather(n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    date = pd.date_range("2021-01-01", periods=n, freq="h")
    t = np.arange(n)
    data = {}
    for i in range(20):
        data[f"feature_{i + 1:02d}"] = (
            np.sin(2 * np.pi * t / (24 * (i % 5 + 1)))
            + 0.03 * i
            + rng.normal(0, 0.05, size=n)
        )
    data["OT"] = (
        15
        + 5 * np.sin(2 * np.pi * t / 24)
        + 1.5 * np.sin(2 * np.pi * t / (24 * 7))
        + rng.normal(0, 0.35, size=n)
    )
    return pd.DataFrame({"date": date, **data})


def read_weather_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "date" not in df.columns:
        df.insert(0, "date", pd.date_range("2021-01-01", periods=len(df), freq="h"))

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        df["date"] = pd.date_range("2021-01-01", periods=len(df), freq="h")

    numeric_cols = [c for c in df.columns if c != "date"]
    if not numeric_cols:
        raise ValueError("CSV must contain at least one numeric column besides `date`.")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("CSV has no valid numeric rows after cleaning.")

    return df


def load_active_dataframe() -> Tuple[pd.DataFrame, str, Path]:
    """Load user CSV first, then legacy CALF CSV, then bundled sample."""
    for path, label in [
        (USER_CSV_PATH, "Uploaded CSV"),
        (LEGACY_CSV_PATH, "CALF project CSV"),
        (SAMPLE_CSV_PATH, "Bundled sample CSV"),
    ]:
        if path.exists():
            return read_weather_csv(path), label, path

    return create_synthetic_weather(), "Synthetic demo data", DATA_DIR / "synthetic_weather.csv"


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]


def dataset_custom_order(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, List[str], int]:
    """Match the common time-series `date + features + target` ordering."""
    features = numeric_feature_columns(df)
    if target not in features:
        raise ValueError(f"Target column `{target}` is not numeric or does not exist.")
    ordered_features = [c for c in features if c != target] + [target]
    ordered = df[["date"] + ordered_features].copy()
    return ordered, ordered_features, ordered_features.index(target)


def fit_train_scaler(values: np.ndarray) -> SimpleStandardScaler:
    n_train = max(1, int(len(values) * 0.7))
    return SimpleStandardScaler().fit(values[:n_train])


def inverse_target(values: np.ndarray, scaler: SimpleStandardScaler, n_features: int, target_idx: int) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    dummy = np.zeros((len(flat), n_features), dtype=float)
    dummy[:, target_idx] = flat
    restored = scaler.inverse_transform(dummy)
    return restored[:, target_idx]


def extract_target_series(arr: np.ndarray, target_idx: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        channel = target_idx if arr.shape[-1] > target_idx else 0
        return arr[0, :, channel]
    if arr.ndim == 2:
        channel = target_idx if arr.shape[-1] > target_idx else 0
        return arr[:, channel]
    if arr.ndim == 1:
        return arr
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    eps = 1e-8
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))
    mspe = float(np.mean(((y_true - y_pred) / (np.abs(y_true) + eps)) ** 2))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "MSPE": mspe}


def forecast_chart(history: np.ndarray, pred: np.ndarray, true: Optional[np.ndarray] = None) -> pd.DataFrame:
    history = np.asarray(history, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    total = len(history) + len(pred)
    out = pd.DataFrame(index=np.arange(total))
    out["Observed history"] = np.nan
    out.loc[: len(history) - 1, "Observed history"] = history
    out["Forecast"] = np.nan
    out.loc[len(history) : total - 1, "Forecast"] = pred
    if true is not None:
        true = np.asarray(true, dtype=float).reshape(-1)
        out["Ground truth"] = np.nan
        out.loc[len(history) : len(history) + len(true) - 1, "Ground truth"] = true[: len(pred)]
    out.index.name = "time_step"
    return out


def show_metrics(metrics: Dict[str, float]) -> None:
    cols = st.columns(len(METRIC_NAMES))
    for col, name in zip(cols, METRIC_NAMES):
        value = metrics.get(name)
        col.metric(name, "—" if value is None else f"{value:.5f}")


# ============================================================
# Forecast engines
# ============================================================

@st.cache_resource(show_spinner=False)
def load_onnx_session(path: str):
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def build_onnx_input(df: pd.DataFrame, cfg: ForecastConfig) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], SimpleStandardScaler, List[str], int]:
    ordered, features, target_idx = dataset_custom_order(df, cfg.target)
    values = ordered[features].to_numpy(dtype=float)

    if len(values) < cfg.seq_len:
        raise ValueError(f"Need at least {cfg.seq_len} rows, but the CSV has only {len(values)} rows.")

    scaler = fit_train_scaler(values)
    values_scaled = scaler.transform(values)

    if cfg.backtest:
        needed = cfg.seq_len + cfg.pred_len
        if len(values_scaled) < needed:
            raise ValueError(
                f"Backtest needs at least seq_len + pred_len = {needed} rows, "
                f"but the CSV has only {len(values_scaled)} rows."
            )
        x = values_scaled[-needed : -cfg.pred_len]
        history_raw = values[-needed : -cfg.pred_len, target_idx]
        true_raw = values[-cfg.pred_len :, target_idx]
    else:
        x = values_scaled[-cfg.seq_len :]
        history_raw = values[-cfg.seq_len :, target_idx]
        true_raw = None

    return x[np.newaxis, :, :].astype(np.float32), history_raw, true_raw, scaler, features, target_idx


def run_onnx_forecast(df: pd.DataFrame, cfg: ForecastConfig) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    x, history_raw, true_raw, scaler, features, target_idx = build_onnx_input(df, cfg)

    session = load_onnx_session(str(ONNX_PATH.resolve()))
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    expected_features = input_meta.shape[-1]
    if isinstance(expected_features, int) and expected_features != x.shape[-1]:
        raise ValueError(
            f"The ONNX model expects {expected_features} features, but the current CSV has {x.shape[-1]}. "
            "Please use the same Weather CSV format used during model export."
        )

    pred_scaled = session.run([output_meta.name], {input_meta.name: x})[0]
    pred_target_scaled = extract_target_series(pred_scaled, target_idx)
    pred_raw = inverse_target(pred_target_scaled, scaler, len(features), target_idx)
    pred_raw = pred_raw[: cfg.pred_len]

    metrics = calculate_metrics(true_raw, pred_raw) if true_raw is not None else {}

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULT_DIR / "input.npy", x)
    np.save(RESULT_DIR / "pred.npy", pred_scaled)
    if true_raw is not None:
        true_scaled = np.zeros((1, len(true_raw), len(features)), dtype=np.float32)
        true_scaled[0, :, target_idx] = (true_raw - scaler.mean_[target_idx]) / scaler.scale_[target_idx]
        np.save(RESULT_DIR / "true.npy", true_scaled)
    if metrics:
        np.save(RESULT_DIR / "metrics.npy", np.array([metrics[k] for k in METRIC_NAMES], dtype=float))

    return history_raw, pred_raw, true_raw, metrics


def run_demo_forecast(df: pd.DataFrame, cfg: ForecastConfig) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    y = df[cfg.target].to_numpy(dtype=float)

    if cfg.backtest and len(y) >= cfg.seq_len + cfg.pred_len:
        history = y[-(cfg.seq_len + cfg.pred_len) : -cfg.pred_len]
        true = y[-cfg.pred_len :]
    else:
        history = y[-cfg.seq_len :]
        true = None

    last = float(history[-1])
    local_slope = (float(history[-1]) - float(history[max(0, len(history) - 8)])) / max(1, min(8, len(history) - 1))
    steps = np.arange(1, cfg.pred_len + 1)
    seasonal = 0.05 * np.std(history) * np.sin(2 * np.pi * steps / max(24, cfg.pred_len))
    pred = last + local_slope * steps + seasonal
    metrics = calculate_metrics(true, pred) if true is not None else {}
    return history, pred, true, metrics


# ============================================================
# File import helpers
# ============================================================


def save_uploaded_file(uploaded_file, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as out:
        out.write(uploaded_file.getbuffer())


def import_result_file(name: str, data: bytes) -> Tuple[str, str]:
    destination = RESULT_DIR / Path(name).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return name, str(destination)


def import_zip(uploaded_zip) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    try:
        with zipfile.ZipFile(uploaded_zip) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                filename = Path(member.filename).name
                if not filename or filename.startswith("."):
                    continue
                lower = filename.lower()
                data = zf.read(member)
                if lower == "weather.csv":
                    USER_CSV_PATH.write_bytes(data)
                    rows.append((filename, "Imported", str(USER_CSV_PATH)))
                elif lower.endswith(".onnx"):
                    ONNX_PATH.write_bytes(data)
                    rows.append((filename, "Imported", str(ONNX_PATH)))
                elif lower in RECOGNIZED_RESULT_FILES:
                    _, dest = import_result_file(filename, data)
                    rows.append((filename, "Imported", dest))
                else:
                    rows.append((filename, "Skipped", "Not needed by the clean UI"))
    except Exception as exc:
        rows.append((uploaded_zip.name, "Error", str(exc)))
    return rows


def import_uploads(uploaded_files: Iterable) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    for uploaded in uploaded_files:
        lower = uploaded.name.lower()
        try:
            if lower.endswith(".zip"):
                rows.extend(import_zip(uploaded))
            elif lower == "weather.csv" or lower.endswith(".csv"):
                save_uploaded_file(uploaded, USER_CSV_PATH)
                rows.append((uploaded.name, "Imported", str(USER_CSV_PATH)))
            elif lower.endswith(".onnx"):
                save_uploaded_file(uploaded, ONNX_PATH)
                rows.append((uploaded.name, "Imported", str(ONNX_PATH)))
            elif lower in RECOGNIZED_RESULT_FILES:
                _, dest = import_result_file(uploaded.name, uploaded.getbuffer().tobytes())
                rows.append((uploaded.name, "Imported", dest))
            else:
                rows.append((uploaded.name, "Skipped", "Only CSV, ONNX, NPY, and ZIP files are used here"))
        except Exception as exc:
            rows.append((uploaded.name, "Error", str(exc)))
    return rows


def load_saved_arrays() -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for name in RECOGNIZED_RESULT_FILES:
        path = RESULT_DIR / name
        if path.exists():
            arrays[name.replace(".npy", "")] = np.load(path)
    return arrays


# ============================================================
# UI helpers
# ============================================================


def status_text(path: Path) -> str:
    return "Ready" if path.exists() else "Missing"


def sidebar_settings(df: pd.DataFrame) -> ForecastConfig:
    features = numeric_feature_columns(df)
    default_target = "OT" if "OT" in features else features[-1]

    with st.sidebar:
        st.header("Settings")
        target = st.selectbox("Target", features, index=features.index(default_target))
        seq_len = st.number_input("Input length", min_value=12, max_value=720, value=96, step=12)
        pred_len = st.number_input("Forecast horizon", min_value=1, max_value=720, value=96, step=12)
        backtest = st.toggle("Backtest with ground truth", value=True)

        with st.expander("Advanced"):
            inverse_scale = st.checkbox("Inverse-scale saved .npy arrays", value=True)
            st.caption("Disable this only when your saved arrays are already in the original data scale.")

    return ForecastConfig(
        target=target,
        seq_len=int(seq_len),
        pred_len=int(pred_len),
        backtest=bool(backtest),
        inverse_scale_saved_arrays=bool(inverse_scale),
    )


def header() -> None:
    st.title("🌦️ CALF Weather Forecast")
    st.markdown(
        "Upload a Weather CSV and an exported ONNX model, then run forecasting in a clean 3-step interface."
    )


def show_status_cards(df: pd.DataFrame, data_label: str, data_path: Path, cfg: ForecastConfig) -> None:
    arrays = load_saved_arrays()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data", data_label)
    c2.metric("Rows", f"{len(df):,}")
    c3.metric("ONNX model", status_text(ONNX_PATH))
    c4.metric("Saved results", "Ready" if {"pred", "true"}.issubset(arrays) else "Optional")

    with st.expander("Current setup", expanded=False):
        st.write(f"**Data path:** `{data_path}`")
        st.write(f"**Target:** `{cfg.target}`")
        st.write(f"**Input length:** `{cfg.seq_len}`")
        st.write(f"**Forecast horizon:** `{cfg.pred_len}`")
        st.write(f"**ONNX path:** `{ONNX_PATH}`")


def display_forecast_result(history: np.ndarray, pred: np.ndarray, true: Optional[np.ndarray], metrics: Dict[str, float]) -> None:
    if metrics:
        st.subheader("Metrics")
        show_metrics(metrics)
    else:
        st.info("No ground truth was used, so metrics are not available for this run.")

    st.subheader("Forecast chart")
    chart_df = forecast_chart(history, pred, true)
    st.line_chart(chart_df, use_container_width=True)

    out = pd.DataFrame({"step": np.arange(1, len(pred) + 1), "forecast": pred})
    if true is not None:
        n = min(len(true), len(pred))
        out = out.iloc[:n].copy()
        out["ground_truth"] = true[:n]
        out["absolute_error"] = np.abs(out["ground_truth"] - out["forecast"])

    st.subheader("Forecast values")
    st.dataframe(out, use_container_width=True, hide_index=True)
    st.download_button(
        "Download forecast CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="calf_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )


def saved_results_view(df: pd.DataFrame, cfg: ForecastConfig) -> None:
    arrays = load_saved_arrays()
    if not arrays:
        st.info("No saved result arrays found yet. Upload `pred.npy`, `true.npy`, `input.npy`, and optionally `metrics.npy` in Step 1.")
        return

    st.write("Loaded files:", ", ".join(f"`{k}.npy`" for k in sorted(arrays)))

    if "pred" not in arrays or "true" not in arrays:
        st.warning("To visualize saved results, please provide at least `pred.npy` and `true.npy`.")
        return

    pred_arr = arrays["pred"]
    true_arr = arrays["true"]
    input_arr = arrays.get("input")

    n_samples = pred_arr.shape[0] if pred_arr.ndim == 3 else 1
    sample_idx = 0
    if n_samples > 1:
        sample_idx = st.slider("Sample", 0, n_samples - 1, 0)

    _, features, target_idx = dataset_custom_order(df, cfg.target)
    values = dataset_custom_order(df, cfg.target)[0][features].to_numpy(dtype=float)
    scaler = fit_train_scaler(values)

    def pick_sample(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 3:
            return arr[sample_idx : sample_idx + 1]
        return arr

    pred = extract_target_series(pick_sample(pred_arr), target_idx)
    true = extract_target_series(pick_sample(true_arr), target_idx)

    if input_arr is not None:
        history = extract_target_series(pick_sample(input_arr), target_idx)
    else:
        history = df[cfg.target].to_numpy(dtype=float)[-cfg.seq_len :]

    if cfg.inverse_scale_saved_arrays:
        pred = inverse_target(pred, scaler, len(features), target_idx)
        true = inverse_target(true, scaler, len(features), target_idx)
        if input_arr is not None:
            history = inverse_target(history, scaler, len(features), target_idx)

    metrics = calculate_metrics(true, pred)
    display_forecast_result(history, pred, true, metrics)

    with st.expander("Array shapes", expanded=False):
        st.json({name: list(value.shape) for name, value in arrays.items()})


# ============================================================
# Main app
# ============================================================


def main() -> None:
    ensure_dirs()
    header()

    try:
        df, data_label, data_path = load_active_dataframe()
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

    cfg = sidebar_settings(df)
    show_status_cards(df, data_label, data_path, cfg)

    step1, step2, step3 = st.tabs(["1. Upload", "2. Forecast", "3. Results & data"])

    with step1:
        st.subheader("Upload files")
        st.markdown(
            """
            Use this page for only the files needed by the app:

            - `weather.csv` for real Weather data
            - `calf_weather_forecast.onnx` for real CALF ONNX forecasting
            - `pred.npy`, `true.npy`, `input.npy`, `metrics.npy` for saved-result visualization
            - `.zip` containing any of the files above
            """
        )

        uploads = st.file_uploader(
            "Choose files",
            type=["csv", "onnx", "npy", "zip"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        col_a, col_b = st.columns([1, 2])
        with col_a:
            import_clicked = st.button("Import files", type="primary", use_container_width=True)
        with col_b:
            st.markdown('<p class="small-note">After importing, the page reloads automatically with the new data/model status.</p>', unsafe_allow_html=True)

        if import_clicked:
            if not uploads:
                st.warning("Please choose at least one file first.")
            else:
                rows = import_uploads(uploads)
                st.dataframe(pd.DataFrame(rows, columns=["File", "Status", "Destination / note"]), use_container_width=True, hide_index=True)
                if any(row[1] == "Imported" for row in rows):
                    st.success("Import completed. Go to Step 2 to run forecasting or Step 3 to view saved results.")

        st.divider()
        st.subheader("Current files")
        file_status = pd.DataFrame(
            [
                ["Weather CSV", USER_CSV_PATH.exists() or LEGACY_CSV_PATH.exists() or SAMPLE_CSV_PATH.exists(), str(data_path)],
                ["ONNX model", ONNX_PATH.exists(), str(ONNX_PATH)],
                ["pred.npy", (RESULT_DIR / "pred.npy").exists(), str(RESULT_DIR / "pred.npy")],
                ["true.npy", (RESULT_DIR / "true.npy").exists(), str(RESULT_DIR / "true.npy")],
                ["input.npy", (RESULT_DIR / "input.npy").exists(), str(RESULT_DIR / "input.npy")],
                ["metrics.npy", (RESULT_DIR / "metrics.npy").exists(), str(RESULT_DIR / "metrics.npy")],
            ],
            columns=["Item", "Ready", "Path"],
        )
        st.dataframe(file_status, use_container_width=True, hide_index=True)

    with step2:
        st.subheader("Run forecast")

        engine = "ONNX model" if ONNX_PATH.exists() else "Demo forecast"
        engine = st.radio(
            "Forecast engine",
            options=["ONNX model", "Demo forecast"],
            index=0 if ONNX_PATH.exists() else 1,
            horizontal=True,
            help="Use ONNX for real model inference. Demo forecast is only for checking the UI.",
        )

        if engine == "ONNX model" and not ONNX_PATH.exists():
            st.warning("No ONNX model found. Upload `calf_weather_forecast.onnx` in Step 1, or use Demo forecast.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Target", cfg.target)
        c2.metric("Input length", cfg.seq_len)
        c3.metric("Horizon", cfg.pred_len)

        run_clicked = st.button("Run forecast", type="primary", use_container_width=True)

        if run_clicked:
            try:
                if engine == "ONNX model":
                    if not ONNX_PATH.exists():
                        raise FileNotFoundError("ONNX model is missing. Please upload it in Step 1.")
                    with st.spinner("Running ONNX forecast..."):
                        history, pred, true, metrics = run_onnx_forecast(df, cfg)
                    st.success("Forecast completed with ONNX model.")
                else:
                    history, pred, true, metrics = run_demo_forecast(df, cfg)
                    st.warning("Demo forecast completed. This is not real CALF inference.")

                display_forecast_result(history, pred, true, metrics)
            except Exception as exc:
                st.exception(exc)

    with step3:
        st.subheader("Saved results")
        saved_results_view(df, cfg)

        st.divider()
        st.subheader("Data preview")
        preview_cols = ["date", cfg.target] if "date" in df.columns else [cfg.target]
        st.line_chart(df.set_index("date")[[cfg.target]].tail(500), use_container_width=True)
        st.dataframe(df[preview_cols + [c for c in numeric_feature_columns(df) if c != cfg.target][:6]].head(100), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
