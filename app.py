"""Streamlit UI for lightweight CALF Weather forecasting/export packs.

This app supports the lightweight ZIP produced by the corrected notebook:

    CALF_weather_light_export.zip

Supported import contents:
- data/weather.csv
- results/input.npy, pred.npy, true.npy, metrics.npy, sample_indices.npy
- scaler/weather_scaler.npz
- metadata.json
- optional ONNX files such as onnx/nlinear_weather.onnx or onnx/dlinear_weather.onnx

The key fix compared with the old UI is that saved arrays are inverse-scaled with
`scaler/weather_scaler.npz` from the same notebook run instead of refitting a new
scaler with the wrong train split.
"""

from __future__ import annotations

import json
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
SCALER_DIR = APP_DIR / "scaler"

USER_CSV_PATH = DATA_DIR / "weather.csv"
SAMPLE_CSV_PATH = DATA_DIR / "weather_sample.csv"
LEGACY_CSV_PATH = APP_DIR / "CALF" / "datasets" / "weather" / "weather.csv"
ONNX_PATH = WEIGHT_DIR / "calf_weather_forecast.onnx"
SCALER_PATH = SCALER_DIR / "weather_scaler.npz"
METADATA_PATH = APP_DIR / "metadata.json"

METRIC_NAMES = ["MAE", "MSE", "RMSE", "MAPE", "MSPE"]
RESULT_ARRAY_FILES = {"pred.npy", "true.npy", "input.npy", "metrics.npy", "sample_indices.npy"}
REQUIRED_VISUAL_FILES = {"pred", "true"}


# ============================================================
# Styling
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
        arr = np.asarray(values, dtype=np.float64)
        self.mean_ = np.nanmean(arr, axis=0)
        self.scale_ = np.nanstd(arr, axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    @classmethod
    def from_arrays(cls, mean: np.ndarray, scale: np.ndarray) -> "SimpleStandardScaler":
        obj = cls()
        obj.mean_ = np.asarray(mean, dtype=np.float64)
        obj.scale_ = np.asarray(scale, dtype=np.float64)
        obj.scale_ = np.where(obj.scale_ == 0, 1.0, obj.scale_)
        return obj

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale_ + self.mean_


def ensure_dirs() -> None:
    for path in [DATA_DIR, WEIGHT_DIR, RESULT_DIR, SCALER_DIR]:
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


@st.cache_data(show_spinner=False)
def load_export_metadata() -> Dict:
    if not METADATA_PATH.exists():
        return {}
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _npz_scalar_to_python(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr.tolist()


def load_scaler_feature_cols() -> List[str]:
    if not SCALER_PATH.exists():
        return []
    try:
        with np.load(SCALER_PATH, allow_pickle=True) as zf:
            if "feature_cols" in zf:
                return [str(x) for x in zf["feature_cols"].tolist()]
    except Exception:
        return []
    return []


def load_exported_scaler(expected_feature_count: int) -> Optional[SimpleStandardScaler]:
    """Load scaler/weather_scaler.npz from the same notebook export, if valid."""
    if not SCALER_PATH.exists():
        return None

    try:
        with np.load(SCALER_PATH, allow_pickle=True) as zf:
            mean = np.asarray(zf["mean"], dtype=np.float64)
            if "std" in zf:
                scale = np.asarray(zf["std"], dtype=np.float64)
            elif "scale" in zf:
                scale = np.asarray(zf["scale"], dtype=np.float64)
            else:
                return None
    except Exception:
        return None

    if mean.ndim != 1 or scale.ndim != 1 or len(mean) != expected_feature_count or len(scale) != expected_feature_count:
        return None

    return SimpleStandardScaler.from_arrays(mean, scale)


def metadata_train_ratio(default: float = 0.8) -> float:
    meta = load_export_metadata()
    try:
        ratio = float(meta.get("train_ratio", default))
        return ratio if 0 < ratio < 1 else default
    except Exception:
        return default


def dataset_custom_order(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, List[str], int]:
    """Use exported feature order when available; otherwise move target to the last channel."""
    numeric_cols = numeric_feature_columns(df)
    if target not in numeric_cols:
        raise ValueError(f"Target column `{target}` is not numeric or does not exist.")

    meta = load_export_metadata()
    meta_features = meta.get("feature_cols") if isinstance(meta.get("feature_cols"), list) else []
    scaler_features = load_scaler_feature_cols()

    for candidate_features in [meta_features, scaler_features]:
        if candidate_features and all(c in numeric_cols for c in candidate_features):
            ordered_features = [str(c) for c in candidate_features]
            target_idx = ordered_features.index(target) if target in ordered_features else len(ordered_features) - 1
            ordered = df[["date"] + ordered_features].copy()
            return ordered, ordered_features, target_idx

    ordered_features = [c for c in numeric_cols if c != target] + [target]
    ordered = df[["date"] + ordered_features].copy()
    return ordered, ordered_features, ordered_features.index(target)


def fit_train_scaler(values: np.ndarray) -> SimpleStandardScaler:
    # Corrected default: notebook uses 80/10/10, not the old 70% split.
    train_ratio = metadata_train_ratio(default=0.8)
    n_train = max(1, int(len(values) * train_ratio))
    return SimpleStandardScaler().fit(values[:n_train])


def get_active_scaler(values: np.ndarray, features: List[str]) -> Tuple[SimpleStandardScaler, str]:
    exported = load_exported_scaler(expected_feature_count=len(features))
    if exported is not None:
        return exported, "exported scaler"
    return fit_train_scaler(values), f"refit {metadata_train_ratio(default=0.8):.0%} train scaler"


def inverse_target(values: np.ndarray, scaler: SimpleStandardScaler, n_features: int, target_idx: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    dummy = np.zeros((len(flat), n_features), dtype=np.float64)
    channel = target_idx if 0 <= target_idx < n_features else n_features - 1
    dummy[:, channel] = flat
    restored = scaler.inverse_transform(dummy)
    return restored[:, channel]


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
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return {}
    y_true, y_pred = y_true[:n], y_pred[:n]
    eps = 1e-8
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))
    mspe = float(np.mean(((y_true - y_pred) / (np.abs(y_true) + eps)) ** 2))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "MSPE": mspe}


def metric_dict_from_array(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr).reshape(-1)
    if arr.size < len(METRIC_NAMES):
        return {}
    return {name: float(arr[i]) for i, name in enumerate(METRIC_NAMES)}


def forecast_chart(history: np.ndarray, pred: np.ndarray, true: Optional[np.ndarray] = None) -> pd.DataFrame:
    history = np.asarray(history, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    total = len(history) + len(pred)
    out = pd.DataFrame(index=np.arange(total))
    out["Observed history"] = np.nan
    out.loc[: len(history) - 1, "Observed history"] = history
    out["Forecast"] = np.nan
    out.loc[len(history): total - 1, "Forecast"] = pred
    if true is not None:
        true = np.asarray(true, dtype=np.float64).reshape(-1)
        out["Ground truth"] = np.nan
        out.loc[len(history): len(history) + min(len(true), len(pred)) - 1, "Ground truth"] = true[: len(pred)]
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


def build_onnx_input(df: pd.DataFrame, cfg: ForecastConfig) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], SimpleStandardScaler, List[str], int, str]:
    ordered, features, target_idx = dataset_custom_order(df, cfg.target)
    values = ordered[features].to_numpy(dtype=np.float64)

    if len(values) < cfg.seq_len:
        raise ValueError(f"Need at least {cfg.seq_len} rows, but the CSV has only {len(values)} rows.")

    scaler, scaler_source = get_active_scaler(values, features)
    values_scaled = scaler.transform(values)

    if cfg.backtest:
        needed = cfg.seq_len + cfg.pred_len
        if len(values_scaled) < needed:
            raise ValueError(
                f"Backtest needs at least seq_len + pred_len = {needed} rows, "
                f"but the CSV has only {len(values_scaled)} rows."
            )
        x = values_scaled[-needed: -cfg.pred_len]
        history_raw = values[-needed: -cfg.pred_len, target_idx]
        true_raw = values[-cfg.pred_len:, target_idx]
    else:
        x = values_scaled[-cfg.seq_len:]
        history_raw = values[-cfg.seq_len:, target_idx]
        true_raw = None

    return x[np.newaxis, :, :].astype(np.float32), history_raw, true_raw, scaler, features, target_idx, scaler_source


def _validate_onnx_shape(session, x: np.ndarray) -> None:
    input_meta = session.get_inputs()[0]
    shape = list(input_meta.shape)

    if len(shape) >= 2 and isinstance(shape[1], int) and shape[1] != x.shape[1]:
        raise ValueError(
            f"The ONNX model expects seq_len={shape[1]}, but the app is using seq_len={x.shape[1]}. "
            "Use the seq_len from metadata.json, usually 336 for this notebook export."
        )

    if len(shape) >= 3 and isinstance(shape[2], int) and shape[2] != x.shape[2]:
        raise ValueError(
            f"The ONNX model expects {shape[2]} features, but the current CSV has {x.shape[2]}. "
            "Please use the same Weather CSV format used during model export."
        )


def run_onnx_forecast(df: pd.DataFrame, cfg: ForecastConfig) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    x, history_raw, true_raw, scaler, features, target_idx, scaler_source = build_onnx_input(df, cfg)

    session = load_onnx_session(str(ONNX_PATH.resolve()))
    _validate_onnx_shape(session, x)
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    pred_scaled = session.run([output_meta.name], {input_meta.name: x})[0]
    pred_target_scaled = extract_target_series(pred_scaled, target_idx)
    pred_raw = inverse_target(pred_target_scaled, scaler, len(features), target_idx)
    pred_raw = pred_raw[: cfg.pred_len]

    metrics = calculate_metrics(true_raw, pred_raw) if true_raw is not None else {}

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULT_DIR / "input.npy", x.astype(np.float16))
    np.save(RESULT_DIR / "pred.npy", pred_scaled.astype(np.float16 if pred_scaled.dtype.kind == "f" else pred_scaled.dtype))
    if true_raw is not None:
        true_scaled = np.zeros((1, len(true_raw), len(features)), dtype=np.float32)
        true_scaled[0, :, target_idx] = (true_raw - scaler.mean_[target_idx]) / scaler.scale_[target_idx]
        np.save(RESULT_DIR / "true.npy", true_scaled.astype(np.float16))
    if metrics:
        np.save(RESULT_DIR / "metrics.npy", np.array([metrics[k] for k in METRIC_NAMES], dtype=np.float32))

    st.caption(f"ONNX input used `{scaler_source}`.")
    return history_raw, pred_raw, true_raw, metrics


def run_demo_forecast(df: pd.DataFrame, cfg: ForecastConfig) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    y = df[cfg.target].to_numpy(dtype=np.float64)

    if cfg.backtest and len(y) >= cfg.seq_len + cfg.pred_len:
        history = y[-(cfg.seq_len + cfg.pred_len): -cfg.pred_len]
        true = y[-cfg.pred_len:]
    else:
        history = y[-cfg.seq_len:]
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


def clear_result_artifacts() -> None:
    for name in RESULT_ARRAY_FILES:
        path = RESULT_DIR / name
        if path.exists():
            path.unlink()
    if SCALER_PATH.exists():
        SCALER_PATH.unlink()
    if METADATA_PATH.exists():
        METADATA_PATH.unlink()
    load_export_metadata.clear()


def clear_all_imported_artifacts() -> None:
    clear_result_artifacts()
    for path in [USER_CSV_PATH, ONNX_PATH]:
        if path.exists():
            path.unlink()
    load_onnx_session.clear()


def import_result_file(name: str, data: bytes) -> Tuple[str, str]:
    destination = RESULT_DIR / Path(name).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return name, str(destination)


def import_zip(uploaded_zip) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    try:
        with zipfile.ZipFile(uploaded_zip) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            basenames = {Path(m.filename).name.lower() for m in members}

            # Avoid mixing old pred/true/input/scaler with a new lightweight export.
            if basenames & (RESULT_ARRAY_FILES | {"weather_scaler.npz", "metadata.json"}):
                clear_result_artifacts()

            for member in members:
                filename = Path(member.filename).name
                if not filename or filename.startswith("."):
                    continue

                lower = filename.lower()
                data = zf.read(member)

                if lower == "weather.csv":
                    USER_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
                    USER_CSV_PATH.write_bytes(data)
                    rows.append((filename, "Imported", str(USER_CSV_PATH)))
                elif lower.endswith(".onnx"):
                    # Store any imported ONNX as the active model, regardless of original filename.
                    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
                    ONNX_PATH.write_bytes(data)
                    load_onnx_session.clear()
                    rows.append((filename, "Imported as active ONNX", str(ONNX_PATH)))
                elif lower in RESULT_ARRAY_FILES:
                    _, dest = import_result_file(filename, data)
                    rows.append((filename, "Imported", dest))
                elif lower == "weather_scaler.npz":
                    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
                    SCALER_PATH.write_bytes(data)
                    rows.append((filename, "Imported", str(SCALER_PATH)))
                elif lower == "metadata.json":
                    METADATA_PATH.write_bytes(data)
                    load_export_metadata.clear()
                    rows.append((filename, "Imported", str(METADATA_PATH)))
                else:
                    rows.append((filename, "Skipped", "Not needed by this lightweight UI"))
    except Exception as exc:
        rows.append((getattr(uploaded_zip, "name", "uploaded.zip"), "Error", str(exc)))
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
                load_onnx_session.clear()
                rows.append((uploaded.name, "Imported as active ONNX", str(ONNX_PATH)))
            elif lower in RESULT_ARRAY_FILES:
                _, dest = import_result_file(uploaded.name, uploaded.getbuffer().tobytes())
                rows.append((uploaded.name, "Imported", dest))
            elif lower == "weather_scaler.npz":
                save_uploaded_file(uploaded, SCALER_PATH)
                rows.append((uploaded.name, "Imported", str(SCALER_PATH)))
            elif lower == "metadata.json":
                save_uploaded_file(uploaded, METADATA_PATH)
                load_export_metadata.clear()
                rows.append((uploaded.name, "Imported", str(METADATA_PATH)))
            else:
                rows.append((uploaded.name, "Skipped", "Only CSV, ONNX, NPY, NPZ, JSON, and ZIP files are used here"))
        except Exception as exc:
            rows.append((uploaded.name, "Error", str(exc)))
    return rows


def load_saved_arrays() -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for name in RESULT_ARRAY_FILES:
        path = RESULT_DIR / name
        if path.exists():
            try:
                arrays[name.replace(".npy", "")] = np.load(path, allow_pickle=False)
            except Exception:
                pass
    return arrays


# ============================================================
# UI helpers
# ============================================================


def status_text(path: Path) -> str:
    return "Ready" if path.exists() else "Missing"


def sidebar_settings(df: pd.DataFrame) -> ForecastConfig:
    features = numeric_feature_columns(df)
    meta = load_export_metadata()

    meta_target = str(meta.get("target", "")) if meta.get("target") is not None else ""
    default_target = meta_target if meta_target in features else ("OT" if "OT" in features else features[-1])
    default_seq = int(meta.get("seq_len", 336) or 336)
    default_pred = int(meta.get("pred_len", 96) or 96)

    with st.sidebar:
        st.header("Settings")
        target = st.selectbox("Target", features, index=features.index(default_target))
        seq_len = st.number_input("Input length", min_value=12, max_value=720, value=default_seq, step=12)
        pred_len = st.number_input("Forecast horizon", min_value=1, max_value=720, value=default_pred, step=12)
        backtest = st.toggle("Backtest with ground truth", value=True)

        with st.expander("Advanced"):
            inverse_scale = st.checkbox("Inverse-scale saved .npy arrays", value=True)
            st.caption("Keep this ON for the lightweight export ZIP. Turn it off only if arrays are already in original scale.")

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
        "Upload the lightweight export ZIP from the notebook, or upload a Weather CSV and ONNX model manually."
    )


def show_status_cards(df: pd.DataFrame, data_label: str, data_path: Path, cfg: ForecastConfig) -> None:
    arrays = load_saved_arrays()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Data", data_label)
    c2.metric("Rows", f"{len(df):,}")
    c3.metric("ONNX model", status_text(ONNX_PATH))
    c4.metric("Scaler", status_text(SCALER_PATH))
    c5.metric("Saved results", "Ready" if REQUIRED_VISUAL_FILES.issubset(arrays) else "Optional")

    with st.expander("Current setup", expanded=False):
        st.write(f"**Data path:** `{data_path}`")
        st.write(f"**Target:** `{cfg.target}`")
        st.write(f"**Input length:** `{cfg.seq_len}`")
        st.write(f"**Forecast horizon:** `{cfg.pred_len}`")
        st.write(f"**ONNX path:** `{ONNX_PATH}`")
        st.write(f"**Scaler path:** `{SCALER_PATH}`")
        st.write(f"**Metadata path:** `{METADATA_PATH}`")
        meta = load_export_metadata()
        if meta:
            st.json({k: meta.get(k) for k in ["dataset", "seq_len", "pred_len", "target", "target_idx", "train_ratio", "max_result_samples"] if k in meta})


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
        st.info("No saved result arrays found yet. Upload the full `CALF_weather_light_export.zip` in Step 1.")
        return

    st.write("Loaded files:", ", ".join(f"`{k}.npy`" for k in sorted(arrays)))

    if not REQUIRED_VISUAL_FILES.issubset(arrays):
        st.warning("To visualize saved results, please provide at least `pred.npy` and `true.npy`.")
        return

    pred_arr = arrays["pred"]
    true_arr = arrays["true"]
    input_arr = arrays.get("input")
    sample_indices = arrays.get("sample_indices")

    n_samples = pred_arr.shape[0] if pred_arr.ndim == 3 else 1
    sample_idx = 0
    if n_samples > 1:
        sample_idx = st.slider("Sample", 0, n_samples - 1, 0)
        if sample_indices is not None and len(sample_indices) > sample_idx:
            st.caption(f"Original test-window index from export: `{int(sample_indices[sample_idx])}`")

    ordered, features, target_idx = dataset_custom_order(df, cfg.target)
    values = ordered[features].to_numpy(dtype=np.float64)
    scaler, scaler_source = get_active_scaler(values, features)

    def pick_sample(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 3:
            return arr[sample_idx: sample_idx + 1]
        return arr

    pred = extract_target_series(pick_sample(pred_arr), target_idx)
    true = extract_target_series(pick_sample(true_arr), target_idx)

    if input_arr is not None:
        history = extract_target_series(pick_sample(input_arr), target_idx)
    else:
        history = df[cfg.target].to_numpy(dtype=np.float64)[-cfg.seq_len:]

    if cfg.inverse_scale_saved_arrays:
        pred = inverse_target(pred, scaler, len(features), target_idx)
        true = inverse_target(true, scaler, len(features), target_idx)
        if input_arr is not None:
            history = inverse_target(history, scaler, len(features), target_idx)

    sample_metrics = calculate_metrics(true, pred)
    st.caption(f"Saved arrays use `{scaler_source}` for inverse scaling.")
    display_forecast_result(history, pred, true, sample_metrics)

    if "metrics" in arrays:
        exported_metrics = metric_dict_from_array(arrays["metrics"])
        if exported_metrics:
            with st.expander("Original exported metrics.npy", expanded=False):
                show_metrics(exported_metrics)

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
            Best option: upload the full lightweight ZIP from the notebook:

            ```text
            CALF_weather_light_export.zip
            ```

            The app will import `weather.csv`, `weather_scaler.npz`, `metadata.json`, and result arrays together.
            This avoids the old problem where new predictions were mixed with old/scaled files.
            """
        )

        if "last_import_rows" in st.session_state:
            st.success("Last import completed.")
            st.dataframe(
                pd.DataFrame(st.session_state["last_import_rows"], columns=["File", "Status", "Destination / note"]),
                use_container_width=True,
                hide_index=True,
            )

        uploads = st.file_uploader(
            "Choose files",
            type=["csv", "onnx", "npy", "npz", "json", "zip"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            import_clicked = st.button("Import files", type="primary", use_container_width=True)
        with col_b:
            clear_results_clicked = st.button("Clear old results/scaler", use_container_width=True)
        with col_c:
            clear_all_clicked = st.button("Clear all imported files", use_container_width=True)

        if clear_results_clicked:
            clear_result_artifacts()
            st.session_state.pop("last_import_rows", None)
            st.success("Old result/scaler/metadata files cleared.")
            st.rerun()

        if clear_all_clicked:
            clear_all_imported_artifacts()
            st.session_state.pop("last_import_rows", None)
            st.success("All imported data/model/result files cleared.")
            st.rerun()

        if import_clicked:
            if not uploads:
                st.warning("Please choose at least one file first.")
            else:
                rows = import_uploads(uploads)
                st.session_state["last_import_rows"] = rows
                if any(row[1].startswith("Imported") for row in rows):
                    st.rerun()
                else:
                    st.dataframe(pd.DataFrame(rows, columns=["File", "Status", "Destination / note"]), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Current files")
        file_status = pd.DataFrame(
            [
                ["Weather CSV", USER_CSV_PATH.exists() or LEGACY_CSV_PATH.exists() or SAMPLE_CSV_PATH.exists(), str(data_path)],
                ["ONNX model", ONNX_PATH.exists(), str(ONNX_PATH)],
                ["weather_scaler.npz", SCALER_PATH.exists(), str(SCALER_PATH)],
                ["metadata.json", METADATA_PATH.exists(), str(METADATA_PATH)],
                ["pred.npy", (RESULT_DIR / "pred.npy").exists(), str(RESULT_DIR / "pred.npy")],
                ["true.npy", (RESULT_DIR / "true.npy").exists(), str(RESULT_DIR / "true.npy")],
                ["input.npy", (RESULT_DIR / "input.npy").exists(), str(RESULT_DIR / "input.npy")],
                ["metrics.npy", (RESULT_DIR / "metrics.npy").exists(), str(RESULT_DIR / "metrics.npy")],
                ["sample_indices.npy", (RESULT_DIR / "sample_indices.npy").exists(), str(RESULT_DIR / "sample_indices.npy")],
            ],
            columns=["Item", "Ready", "Path"],
        )
        st.dataframe(file_status, use_container_width=True, hide_index=True)

    with step2:
        st.subheader("Run forecast")

        engine = st.radio(
            "Forecast engine",
            options=["ONNX model", "Demo forecast"],
            index=0 if ONNX_PATH.exists() else 1,
            horizontal=True,
            help="Use ONNX for real model inference. Demo forecast is only for checking the UI.",
        )

        if engine == "ONNX model" and not ONNX_PATH.exists():
            st.warning("No ONNX model found. Upload an ONNX file in Step 1, or use Demo forecast.")

        if engine == "Demo forecast":
            st.warning("Demo forecast is only a UI sanity check. It is not CALF and can look flat/noisy.")

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
        if "date" in df.columns:
            st.line_chart(df.set_index("date")[[cfg.target]].tail(500), use_container_width=True)
        else:
            st.line_chart(df[[cfg.target]].tail(500), use_container_width=True)
        extra_cols = [c for c in numeric_feature_columns(df) if c != cfg.target][:6]
        st.dataframe(df[preview_cols + extra_cols].head(100), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
