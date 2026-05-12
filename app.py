# app.py — Streamlit UI for CALF Weather Forecasting
# Deploy folder version generated for the corrected CALF Weather notebook.
# --------------------------------------------------
# Run locally:
#   streamlit run app.py
#
# Expected CALF project structure:
#   CALF/
#     run.py
#     models/CALF.py
#     models/GPT2_arch.py
#     exp/exp_long_term_forecasting.py
#     wte_pca_500.pt
#     datasets/weather/weather.csv
#     checkpoints/<setting>/checkpoint.pth
#
# This app has three modes:
#   1) Demo placeholder mode: works even without a real checkpoint.
#   2) ONNX inference mode: uses calf_weather_forecast.onnx with onnxruntime only.
#   3) Legacy real CALF inference mode: uses PyTorch checkpoint.pth when available.

from __future__ import annotations

import os
import re
import sys
import shlex
import shutil
import subprocess
import zipfile
import html as html_lib
import http.cookiejar
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Large-upload helper: write uploaded files in chunks instead of duplicating
# the whole file again with uploaded_file.getvalue().
UPLOAD_COPY_CHUNK_SIZE = 16 * 1024 * 1024  # 16 MB
URL_DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB

def write_uploaded_file(uploaded_file, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    uploaded_file.seek(0)
    with dest.open("wb") as out:
        while True:
            chunk = uploaded_file.read(UPLOAD_COPY_CHUNK_SIZE)
            if not chunk:
                break
            out.write(chunk)
    uploaded_file.seek(0)


# ============================================================
# Streamlit page configuration
# ============================================================

st.set_page_config(
    page_title="CALF Weather Forecasting",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Constants
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_DIR = os.getenv("CALF_PROJECT_DIR", str(APP_DIR / "CALF"))
LOCAL_APP_WEIGHT_PATH = APP_DIR / "weights" / "weather_calf_checkpoint.pth"
LOCAL_APP_ONNX_PATH = APP_DIR / "weights" / "calf_weather_forecast.onnx"
ONNX_FILENAME_ALIASES = {
    "calf_weather_forecast.onnx",
    "calf_forecast.onnx",
    "model.onnx",
}
DEFAULT_DATA_RELATIVE = "datasets/weather/weather.csv"
DEFAULT_WEIGHT_RELATIVE_PLACEHOLDER = (
    "checkpoints/"
    "long_term_forecast_weather_CALF_96_96_CALF_custom_"
    "ftM_sl96_ll0_pl96_dm768_nh4_el2_dl1_df768_fc1_"
    "ebtimeF_dtFalse_no_drive_corrected_gpt6_0/"
    "checkpoint.pth"
)

METRIC_NAMES = ["MAE", "MSE", "RMSE", "MAPE", "MSPE"]


class SimpleStandardScaler:
    """Minimal replacement for sklearn.preprocessing.StandardScaler.

    It keeps the Streamlit Cloud app lightweight while preserving the same
    inverse-scaling behavior needed for result visualization.
    """

    def fit(self, values: np.ndarray) -> "SimpleStandardScaler":
        arr = np.asarray(values, dtype=float)
        self.mean_ = np.nanmean(arr, axis=0)
        self.scale_ = np.nanstd(arr, axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr * self.scale_ + self.mean_


# ============================================================
# Data classes
# ============================================================

@dataclass
class CALFConfig:
    project_dir: Path
    root_path: str
    data_path: str
    model_id: str
    des: str
    target: str
    features: str
    freq: str
    seq_len: int
    label_len: int
    pred_len: int
    enc_in: int
    dec_in: int
    c_out: int
    d_model: int
    n_heads: int
    e_layers: int
    d_layers: int
    d_ff: int
    factor: int
    embed: str
    distil: bool
    batch_size: int
    num_workers: int
    gpt_layers: int
    word_embedding_path: str


# ============================================================
# General helpers
# ============================================================


def show_header() -> None:
    st.title("🌦️ CALF Weather Forecasting Demo")
    st.caption(
        "Streamlit interface for your CALF long-term forecasting model. "
        "Use placeholder mode for UI testing, then switch to ONNX inference after exporting your CALF model."
    )


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def safe_write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def project_is_valid(project_dir: Path) -> Tuple[bool, List[str]]:
    required = [
        project_dir / "run.py",
        project_dir / "models" / "CALF.py",
        project_dir / "models" / "GPT2_arch.py",
        project_dir / "exp" / "exp_long_term_forecasting.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    return len(missing) == 0, missing


def read_weather_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "date" not in df.columns:
        # CALF custom dataset normally expects a date column.
        # We create a synthetic hourly date index so the app can still run.
        df.insert(0, "date", pd.date_range("2020-01-01", periods=len(df), freq="h"))

    # Convert date to string for safe CSV writing and display.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        df["date"] = pd.date_range("2020-01-01", periods=len(df), freq="h")

    numeric_cols = [c for c in df.columns if c != "date"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("CSV has no valid numeric rows after cleaning.")

    return df


def infer_target_and_dim(df: pd.DataFrame, preferred_target: str = "OT") -> Tuple[str, int, List[str]]:
    feature_cols = [c for c in df.columns if c != "date"]
    if not feature_cols:
        raise ValueError("The CSV must contain at least one numeric feature column besides 'date'.")

    target = preferred_target if preferred_target in feature_cols else feature_cols[-1]
    enc_in = len(feature_cols)
    return target, enc_in, feature_cols


def reorder_like_dataset_custom(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, List[str], int]:
    """
    Reconstruct the common Dataset_Custom column order:
      date + all non-target feature columns + target column
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is not in dataframe.")

    cols = list(df.columns)
    cols.remove("date")
    cols.remove(target)
    ordered_df = df[["date"] + cols + [target]].copy()
    feature_cols = list(ordered_df.columns[1:])
    target_idx = feature_cols.index(target)
    return ordered_df, feature_cols, target_idx


def build_setting(cfg: CALFConfig) -> str:
    """
    Matches the common Time-Series-Library/CALF experiment naming format.
    CALF result/checkpoint folders usually use this exact string.
    """
    distil_text = "True" if cfg.distil else "False"
    return (
        f"long_term_forecast_{cfg.model_id}_CALF_custom_"
        f"ft{cfg.features}_sl{cfg.seq_len}_ll{cfg.label_len}_pl{cfg.pred_len}_"
        f"dm{cfg.d_model}_nh{cfg.n_heads}_el{cfg.e_layers}_dl{cfg.d_layers}_"
        f"df{cfg.d_ff}_fc{cfg.factor}_eb{cfg.embed}_dt{distil_text}_"
        f"{cfg.des}_0"
    )


def expected_checkpoint_path(cfg: CALFConfig) -> Path:
    return cfg.project_dir / "checkpoints" / build_setting(cfg) / "checkpoint.pth"


def expected_result_dir(cfg: CALFConfig) -> Path:
    return cfg.project_dir / "results" / build_setting(cfg)


def latest_result_dir(project_dir: Path, model_id: str) -> Optional[Path]:
    root = project_dir / "results"
    if not root.exists():
        return None
    candidates = [p for p in root.glob(f"long_term_forecast_{model_id}_*") if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


# ============================================================
# CALF patch helpers
# ============================================================


def apply_runtime_patches(project_dir: Path) -> List[str]:
    """
    Applies the same safety patches used in your corrected notebook.
    These are compatibility patches only; they do not replace CALF with a dummy model.
    """
    messages: List[str] = []

    calf_path = project_dir / "models" / "CALF.py"
    if calf_path.exists():
        text = safe_read_text(calf_path)

        # Normalize PEFT import.
        if "from peft import" in text:
            text = re.sub(
                r"from peft import .*\n",
                "from peft import LoraConfig, TaskType, get_peft_model\n",
                text,
            )
            messages.append("Patched PEFT import in models/CALF.py")

        # Use feature extraction mode for GPT2Model hidden-state usage.
        if "TaskType.CAUSAL_LM" in text:
            text = text.replace("TaskType.CAUSAL_LM", "TaskType.FEATURE_EXTRACTION")
            messages.append("Changed PEFT task type to FEATURE_EXTRACTION")

        # PyTorch 2.6+ safe loading for PCA embedding.
        if "torch.load(configs.word_embedding_path)" in text:
            text = text.replace(
                "torch.load(configs.word_embedding_path)",
                "torch.load(configs.word_embedding_path, weights_only=False)",
            )
            messages.append("Patched torch.load(..., weights_only=False) for word embedding")

        safe_write_text(calf_path, text)

    gpt_path = project_dir / "models" / "GPT2_arch.py"
    if gpt_path.exists():
        # This wrapper fixes older custom GPT2 forward issues with recent transformers versions.
        gpt_compatible_code = '''import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


class AccustumGPT2Model(GPT2Model):
    """
    Compatibility wrapper for CALF.

    CALF needs the final hidden state and hidden-state list from GPT-2.
    This wrapper delegates to the official GPT2Model.forward and avoids
    old custom-forward incompatibilities with newer transformers versions.
    """

    def forward(self, input_ids=None, labels=None, **kwargs):
        kwargs.pop("labels", None)
        kwargs["output_hidden_states"] = True
        kwargs["output_attentions"] = False
        kwargs["use_cache"] = False
        kwargs["return_dict"] = True
        outputs = super().forward(input_ids=input_ids, **kwargs)
        return outputs.last_hidden_state, outputs.hidden_states
'''
        current = safe_read_text(gpt_path)
        if "class AccustumGPT2Model" not in current or "outputs.last_hidden_state, outputs.hidden_states" not in current:
            safe_write_text(gpt_path, gpt_compatible_code)
            messages.append("Replaced models/GPT2_arch.py with compatible wrapper")

    tools_path = project_dir / "utils" / "tools.py"
    if tools_path.exists():
        text = safe_read_text(tools_path)
        if "np.Inf" in text:
            safe_write_text(tools_path, text.replace("np.Inf", "np.inf"))
            messages.append("Patched np.Inf -> np.inf in utils/tools.py")

    exp_path = project_dir / "exp" / "exp_long_term_forecasting.py"
    if exp_path.exists():
        text = safe_read_text(exp_path)

        old_eval = (
            "        self.model.in_layer.eval()\n"
            "        self.model.out_layer.eval()\n"
            "        self.model.time_proj.eval()\n"
            "        self.model.text_proj.eval()"
        )
        if old_eval in text:
            text = text.replace(old_eval, "        self.model.eval()")
            messages.append("Patched validation/test model.eval()")

        old_train = (
            "        self.model.in_layer.train()\n"
            "        self.model.out_layer.train()\n"
            "        self.model.time_proj.train()\n"
            "        self.model.text_proj.train()"
        )
        if old_train in text:
            text = text.replace(old_train, "        self.model.train()")
            messages.append("Patched model.train()")

        if "inputs = []" not in text:
            text = text.replace(
                "        preds = []\n        trues = []",
                "        preds = []\n        trues = []\n        inputs = []",
            )
            messages.append("Added inputs list in test()")

        if "inputs.append(input_np)" not in text:
            text = text.replace(
                "                pred = outputs_ensemble.detach().cpu().numpy()\n"
                "                true = batch_y.detach().cpu().numpy()\n\n"
                "                preds.append(pred)\n"
                "                trues.append(true)",
                "                pred = outputs_ensemble.detach().cpu().numpy()\n"
                "                true = batch_y.detach().cpu().numpy()\n"
                "                input_np = batch_x.detach().cpu().numpy()\n\n"
                "                preds.append(pred)\n"
                "                trues.append(true)\n"
                "                inputs.append(input_np)",
            )
            messages.append("Patched test() to store input history")

        if "np.save(folder_path + 'input.npy', inputs)" not in text:
            text = text.replace(
                "        preds = np.array(preds)\n"
                "        trues = np.array(trues)\n"
                "        print('test shape:', preds.shape, trues.shape)\n"
                "        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])\n"
                "        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])\n"
                "        print('test shape:', preds.shape, trues.shape)",
                "        preds = np.array(preds)\n"
                "        trues = np.array(trues)\n"
                "        inputs = np.array(inputs)\n"
                "        print('test shape:', preds.shape, trues.shape, inputs.shape)\n"
                "        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])\n"
                "        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])\n"
                "        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])\n"
                "        print('test shape:', preds.shape, trues.shape, inputs.shape)",
            )
            text = text.replace(
                "        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))\n"
                "        np.save(folder_path + 'pred.npy', preds)\n"
                "        np.save(folder_path + 'true.npy', trues)",
                "        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))\n"
                "        np.save(folder_path + 'pred.npy', preds)\n"
                "        np.save(folder_path + 'true.npy', trues)\n"
                "        np.save(folder_path + 'input.npy', inputs)",
            )
            messages.append("Patched test() to save input.npy")

        safe_write_text(exp_path, text)

    if not messages:
        messages.append("No runtime patches were needed")

    return messages


# ============================================================
# Command/inference helpers
# ============================================================


def build_calf_command(cfg: CALFConfig, is_training: int) -> List[str]:
    cmd = [
        sys.executable,
        "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", str(is_training),
        "--root_path", cfg.root_path,
        "--data_path", cfg.data_path,
        "--model_id", cfg.model_id,
        "--model", "CALF",
        "--data", "custom",
        "--features", cfg.features,
        "--target", cfg.target,
        "--freq", cfg.freq,
        "--seq_len", str(cfg.seq_len),
        "--label_len", str(cfg.label_len),
        "--pred_len", str(cfg.pred_len),
        "--enc_in", str(cfg.enc_in),
        "--dec_in", str(cfg.dec_in),
        "--c_out", str(cfg.c_out),
        "--d_model", str(cfg.d_model),
        "--n_heads", str(cfg.n_heads),
        "--e_layers", str(cfg.e_layers),
        "--d_layers", str(cfg.d_layers),
        "--d_ff", str(cfg.d_ff),
        "--factor", str(cfg.factor),
        "--embed", cfg.embed,
        "--des", cfg.des,
        "--itr", "1",
        "--batch_size", str(cfg.batch_size),
        "--num_workers", str(cfg.num_workers),
        "--gpt_layers", str(cfg.gpt_layers),
        "--word_embedding_path", cfg.word_embedding_path,
        "--checkpoints", "./checkpoints/",
    ]

    if cfg.distil:
        cmd.append("--distil")

    return cmd


def copy_checkpoint_to_expected(src_path: Path, dst_path: Path) -> None:
    if not src_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {src_path}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.resolve() != dst_path.resolve():
        shutil.copy2(src_path, dst_path)


def run_subprocess_with_streamlit_logs(cmd: List[str], cwd: Path) -> int:
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"

    log_box = st.empty()
    logs: List[str] = []

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        logs.append(line)
        # Keep the log area readable.
        log_box.code("".join(logs[-120:]), language="text")

    return_code = process.wait()

    log_path = cwd / "streamlit_calf_inference_log.txt"
    log_path.write_text("".join(logs), encoding="utf-8")

    if return_code == 0:
        st.success(f"CALF inference finished. Log saved to: {log_path}")
    else:
        st.error(f"CALF inference failed with exit code {return_code}. Log saved to: {log_path}")

    return return_code


def load_result_arrays(result_dir: Path) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for name in ["metrics", "pred", "true", "input"]:
        path = result_dir / f"{name}.npy"
        if path.exists():
            arrays[name] = np.load(path)
    return arrays



# ============================================================
# Google Drive / direct URL import helpers
# ============================================================


def sanitize_filename(filename: str, fallback: str = "downloaded_file") -> str:
    """Keep only a safe basename for files downloaded from URLs."""
    name = Path(filename or fallback).name.strip().strip('"').strip("'")
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    name = name.strip(" .")
    return name or fallback


def extract_google_drive_file_id(url: str) -> Optional[str]:
    """Extract a file id from common Google Drive sharing URLs."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    if "drive.google.com" not in host and "docs.google.com" not in host:
        return None

    qs = urllib.parse.parse_qs(parsed.query)
    if qs.get("id"):
        return qs["id"][0]

    patterns = [
        r"/file/d/([^/]+)",
        r"/uc\?[^#]*id=([^&]+)",
        r"/open\?[^#]*id=([^&]+)",
        r"/document/d/([^/]+)",
        r"/spreadsheets/d/([^/]+)",
        r"/presentation/d/([^/]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return urllib.parse.unquote(match.group(1))
    return None


def google_drive_download_url(url: str) -> str:
    file_id = extract_google_drive_file_id(url)
    if not file_id:
        return url
    return f"https://drive.google.com/uc?export=download&id={urllib.parse.quote(file_id)}"


def filename_from_content_disposition(content_disposition: Optional[str]) -> Optional[str]:
    if not content_disposition:
        return None

    # RFC 5987 form: filename*=UTF-8''file.ext
    match = re.search(r"filename\*=UTF-8''([^;]+)", content_disposition, flags=re.IGNORECASE)
    if match:
        return urllib.parse.unquote(match.group(1).strip())

    # Common form: filename="file.ext"
    match = re.search(r'filename="?([^";]+)"?', content_disposition, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def _html_attr(tag: str, name: str) -> Optional[str]:
    """Return an HTML attribute value from a single tag string."""
    match = re.search(rf'{name}\s*=\s*(["\'])(.*?)\1', tag, flags=re.IGNORECASE | re.DOTALL)
    return html_lib.unescape(match.group(2)) if match else None


def find_google_drive_confirm_url(html_text: str, current_url: str) -> Optional[str]:
    """Find the second-step download URL from Google Drive's large-file warning page.

    Google Drive changes this page often. This parser handles both the older
    `confirm=` href flow and the newer warning form where input attributes may
    appear in any order.
    """
    text = html_lib.unescape(html_text)

    # Newer Google Drive pages often include a download form with hidden inputs.
    form_match = re.search(
        r'<form[^>]+id=["\']download-form["\'][^>]*>(.*?)</form>',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if form_match:
        form_tag_start = text.rfind("<form", 0, form_match.start(1))
        form_tag_end = text.find(">", form_tag_start)
        form_tag = text[form_tag_start : form_tag_end + 1] if form_tag_start >= 0 and form_tag_end >= 0 else ""
        action = _html_attr(form_tag, "action") or "https://drive.google.com/uc"
        form_html = form_match.group(1)
        params: Dict[str, str] = {}
        for input_match in re.finditer(r"<input\b[^>]*>", form_html, flags=re.IGNORECASE | re.DOTALL):
            input_tag = input_match.group(0)
            name = _html_attr(input_tag, "name")
            if not name:
                continue
            params[name] = _html_attr(input_tag, "value") or ""
        if params:
            return urllib.parse.urljoin(current_url, action) + "?" + urllib.parse.urlencode(params)

    # Older pages expose a direct href containing confirm=...
    href_match = re.search(r'href=["\']([^"\']*confirm=[^"\']*)["\']', text, flags=re.IGNORECASE)
    if href_match:
        href = href_match.group(1).replace("&amp;", "&")
        return urllib.parse.urljoin(current_url, href)

    return None


def google_drive_cookie_confirm_url(url: str, opener: urllib.request.OpenerDirector) -> Optional[str]:
    """Resolve the older Google Drive confirm token stored in cookies."""
    normalized_url = google_drive_download_url(url)
    response = opener.open(
        urllib.request.Request(normalized_url, headers={"User-Agent": "Mozilla/5.0"}),
        timeout=60,
    )
    try:
        for cookie in getattr(opener, "handlers", []):
            pass
        # CookieJar is attached to the HTTPCookieProcessor; easier path is to
        # inspect Set-Cookie headers and extract download_warning tokens.
        set_cookie_headers = response.headers.get_all("Set-Cookie") or []
        confirm_token = None
        for header in set_cookie_headers:
            match = re.search(r"download_warning[^=]*=([^;]+)", header)
            if match:
                confirm_token = match.group(1)
                break
        if not confirm_token:
            return None
        sep = "&" if "?" in normalized_url else "?"
        return normalized_url + sep + "confirm=" + urllib.parse.quote(confirm_token)
    finally:
        response.close()


def open_download_response(url: str, opener: urllib.request.OpenerDirector):
    """Open a URL, including Google Drive's large-file confirmation flow."""
    normalized_url = google_drive_download_url(url)
    request = urllib.request.Request(normalized_url, headers={"User-Agent": "Mozilla/5.0"})
    response = opener.open(request, timeout=60)

    content_type = response.headers.get("Content-Type", "").lower()
    content_disposition = response.headers.get("Content-Disposition")

    # If Google Drive returns an HTML warning/permission page, resolve the real file link.
    if "text/html" in content_type and not content_disposition and "google" in response.geturl().lower():
        html_text = response.read().decode("utf-8", errors="ignore")
        confirm_url = find_google_drive_confirm_url(html_text, response.geturl())
        response.close()
        if not confirm_url:
            raise RuntimeError(
                "Google Drive did not return a direct downloadable file. "
                "This can happen when the file is private, blocked by Google Drive quota, "
                "or Drive shows a large-file warning page that cannot be parsed. "
                "Use the gdown downloader fallback or make sure the file is shared as 'Anyone with the link can view'."
            )
        request = urllib.request.Request(confirm_url, headers={"User-Agent": "Mozilla/5.0"})
        response = opener.open(request, timeout=60)

    return response


def download_google_drive_with_gdown(url: str, download_dir: Path, filename_override: str = "") -> Path:
    """Download Google Drive files with gdown, which handles large-file confirm pages better."""
    file_id = extract_google_drive_file_id(url)
    if not file_id:
        raise RuntimeError("This is not a recognized Google Drive file URL.")

    filename = sanitize_filename(filename_override or "google_drive_download")
    dest = download_dir / filename
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gdown  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "The app needs the `gdown` package for large Google Drive files. "
            "Run `pip install gdown`, or reinstall from the updated requirements.txt."
        ) from exc

    status = st.empty()
    progress = st.progress(0, text=f"Starting Google Drive download: {filename}...")
    candidates = [
        url,
        f"https://drive.google.com/uc?id={urllib.parse.quote(file_id)}",
        f"https://drive.google.com/uc?export=download&id={urllib.parse.quote(file_id)}",
    ]
    last_error: Optional[Exception] = None

    for candidate in candidates:
        try:
            status.write("Downloading from Google Drive. Large files may take several minutes...")
            # fuzzy=True accepts /file/d/.../view URLs. quiet=False gives useful server logs.
            result = gdown.download(candidate, str(dest), quiet=False, fuzzy=True)
            if result and dest.exists() and dest.stat().st_size > 0:
                progress.progress(1.0, text=f"Downloaded {filename}: {dest.stat().st_size / (1024**2):.1f} MB")
                return dest
            if dest.exists() and dest.stat().st_size == 0:
                dest.unlink()
        except Exception as exc:
            last_error = exc
            if dest.exists() and dest.stat().st_size == 0:
                dest.unlink()

    raise RuntimeError(
        "Google Drive download failed. Please check that the file is shared as "
        "'Anyone with the link can view', not restricted to your account. "
        "If it is a very large file, Google Drive may also temporarily block downloads due to quota."
        + (f" Last error: {last_error}" if last_error else "")
    )


def download_file_from_url(url: str, download_dir: Path, filename_override: str = "") -> Path:
    """Download a large file from Google Drive/direct URL by streaming it to disk."""
    download_dir.mkdir(parents=True, exist_ok=True)

    # For Google Drive, prefer gdown because it handles the large-file warning
    # and confirmation flow more reliably than manual urllib parsing.
    if extract_google_drive_file_id(url):
        try:
            return download_google_drive_with_gdown(url, download_dir, filename_override=filename_override)
        except Exception as gdown_exc:
            st.warning(f"gdown failed; trying manual Google Drive/direct downloader. Details: {gdown_exc}")

    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    response = open_download_response(url, opener)
    try:
        cd_name = filename_from_content_disposition(response.headers.get("Content-Disposition"))
        url_name = Path(urllib.parse.urlparse(response.geturl()).path).name
        fallback_name = cd_name or url_name or "downloaded_file"
        filename = sanitize_filename(filename_override or fallback_name)
        dest = download_dir / filename

        total_header = response.headers.get("Content-Length")
        total_bytes = int(total_header) if total_header and total_header.isdigit() else 0

        progress = st.progress(0, text=f"Downloading {filename}...")
        status = st.empty()
        downloaded = 0

        with dest.open("wb") as out:
            while True:
                chunk = response.read(URL_DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total_bytes > 0:
                    progress.progress(
                        min(downloaded / total_bytes, 1.0),
                        text=f"Downloading {filename}: {downloaded / (1024**2):.1f} / {total_bytes / (1024**2):.1f} MB",
                    )
                else:
                    status.write(f"Downloaded {downloaded / (1024**2):.1f} MB...")

        progress.progress(1.0, text=f"Downloaded {filename}: {downloaded / (1024**2):.1f} MB")
        return dest
    finally:
        response.close()


def import_required_zip_path(zip_path: Path, cfg: CALFConfig) -> List[Tuple[str, str, str]]:
    """Import recognized files from a ZIP already stored on disk."""
    rows: List[Tuple[str, str, str]] = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                base = Path(member.filename).name
                if not base or base.startswith(".") or "__MACOSX" in member.filename:
                    continue

                dest = destination_for_imported_file(base, cfg)
                if dest is None:
                    rows.append((member.filename, "Skipped", "Filename not recognized"))
                    continue

                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out, length=UPLOAD_COPY_CHUNK_SIZE)
                rows.append((member.filename, "Imported", str(dest)))
    except zipfile.BadZipFile:
        rows.append((zip_path.name, "Error", "This file is not a valid zip archive"))
    except Exception as exc:
        rows.append((zip_path.name, "Error", str(exc)))
    return rows


def import_downloaded_required_file(src_path: Path, cfg: CALFConfig, import_filename: str = "") -> List[Tuple[str, str, str]]:
    """Import one downloaded file using the same recognized-filename rules as uploads."""
    effective_name = sanitize_filename(import_filename or src_path.name)

    if effective_name.lower().endswith(".zip") or src_path.suffix.lower() == ".zip":
        return import_required_zip_path(src_path, cfg)

    dest = destination_for_imported_file(effective_name, cfg)
    if dest is None:
        return [(effective_name, "Skipped", "Filename not recognized")]

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest)
    return [(effective_name, "Imported", str(dest))]

# ============================================================
# Required-file import helpers
# ============================================================


def destination_for_imported_file(filename: str, cfg: CALFConfig) -> Optional[Path]:
    """Map an uploaded filename to the correct app/CALF destination."""
    base = Path(filename).name
    lower = base.lower()

    if lower == "weather.csv":
        return cfg.project_dir / DEFAULT_DATA_RELATIVE

    if lower == "wte_pca_500.pt":
        return cfg.project_dir / "wte_pca_500.pt"

    if lower in ONNX_FILENAME_ALIASES or lower.endswith(".onnx") or lower.endswith(".onxr"):
        # Store all ONNX/ONXR uploads under a stable filename.
        # If the user accidentally names the ONNX file .onxr, keep the contents but save as .onnx.
        return expected_onnx_path()

    if lower in {"checkpoint.pth", "checkpoint.pt"}:
        return expected_checkpoint_path(cfg)

    if lower in {"metrics.npy", "pred.npy", "true.npy", "input.npy"}:
        return expected_result_dir(cfg) / base

    # Extra model files are kept safely instead of being ignored.
    if lower.endswith((".pth", ".pt")):
        return cfg.project_dir / "streamlit_uploaded_weights" / base

    # Extra .npy files are kept with the result folder. Only pred/true/input/metrics
    # are required for visualization, but this avoids losing useful experiment files.
    if lower.endswith(".npy"):
        return expected_result_dir(cfg) / base

    return None


def save_uploaded_required_file(uploaded_file, cfg: CALFConfig) -> Tuple[str, str, str]:
    dest = destination_for_imported_file(uploaded_file.name, cfg)
    if dest is None:
        return uploaded_file.name, "Skipped", "Filename not recognized"

    write_uploaded_file(uploaded_file, dest)
    return uploaded_file.name, "Imported", str(dest)


def import_required_zip(uploaded_zip, cfg: CALFConfig) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    try:
        uploaded_zip.seek(0)
        with zipfile.ZipFile(uploaded_zip) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                base = Path(member.filename).name
                if not base or base.startswith(".") or "__MACOSX" in member.filename:
                    continue

                dest = destination_for_imported_file(base, cfg)
                if dest is None:
                    rows.append((member.filename, "Skipped", "Filename not recognized"))
                    continue

                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out, length=UPLOAD_COPY_CHUNK_SIZE)
                rows.append((member.filename, "Imported", str(dest)))
    except zipfile.BadZipFile:
        rows.append((uploaded_zip.name, "Error", "This file is not a valid zip archive"))
    except Exception as exc:
        rows.append((uploaded_zip.name, "Error", str(exc)))
    return rows


def required_file_status(cfg: CALFConfig) -> pd.DataFrame:
    result_dir = expected_result_dir(cfg)
    rows = [
        ("weather.csv", cfg.project_dir / DEFAULT_DATA_RELATIVE, "Needed for data preview and ONNX/CALF inference"),
        ("calf_weather_forecast.onnx", expected_onnx_path(), "Needed for ONNX inference without torch"),
        ("wte_pca_500.pt", cfg.project_dir / "wte_pca_500.pt", "Needed only for legacy PyTorch CALF inference"),
        ("checkpoint.pth", expected_checkpoint_path(cfg), "Needed only for legacy PyTorch CALF inference"),
        ("pred.npy", result_dir / "pred.npy", "Needed for saved-result visualization"),
        ("true.npy", result_dir / "true.npy", "Needed for saved-result visualization"),
        ("input.npy", result_dir / "input.npy", "Optional, gives observed history in plots"),
        ("metrics.npy", result_dir / "metrics.npy", "Optional, gives global metrics"),
    ]
    return pd.DataFrame(
        [
            {
                "file": name,
                "status": "OK" if path.exists() else "Missing",
                "destination": str(path),
                "note": note,
            }
            for name, path, note in rows
        ]
    )


# ============================================================
# Plotting / metrics helpers
# ============================================================


def fit_train_scaler(df: pd.DataFrame, target: str) -> Tuple[SimpleStandardScaler, List[str], int]:
    ordered_df, feature_cols, target_idx = reorder_like_dataset_custom(df, target)
    num_train = int(len(ordered_df) * 0.7)
    scaler = SimpleStandardScaler()
    scaler.fit(ordered_df[feature_cols].iloc[:num_train].values)
    return scaler, feature_cols, target_idx


def inverse_target_values(
    values_1d: np.ndarray,
    scaler: SimpleStandardScaler,
    n_features: int,
    target_idx: int,
) -> np.ndarray:
    values_1d = np.asarray(values_1d).reshape(-1)
    dummy = np.zeros((len(values_1d), n_features), dtype=float)
    dummy[:, target_idx] = values_1d
    restored = scaler.inverse_transform(dummy)
    return restored[:, target_idx]


def extract_series(
    arr: np.ndarray,
    sample_idx: int,
    target_idx: int,
) -> np.ndarray:
    """
    Handles arrays with shape:
      [N, L, C], [L, C], or [L]
    """
    arr = np.asarray(arr)
    if arr.ndim == 3:
        sample_idx = max(0, min(sample_idx, arr.shape[0] - 1))
        return arr[sample_idx, :, target_idx]
    if arr.ndim == 2:
        if arr.shape[1] > target_idx:
            return arr[:, target_idx]
        return arr[:, 0]
    if arr.ndim == 1:
        return arr
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def calculate_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    eps = 1e-8
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))
    mspe = float(np.mean(((y_true - y_pred) / (np.abs(y_true) + eps)) ** 2))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "MSPE": mspe}


def display_metrics(metrics: Dict[str, float]) -> None:
    cols = st.columns(len(METRIC_NAMES))
    for col, name in zip(cols, METRIC_NAMES):
        val = metrics.get(name)
        if val is None:
            col.metric(name, "—")
        else:
            col.metric(name, f"{val:.6f}")


def make_forecast_chart_data(
    history: np.ndarray,
    pred: np.ndarray,
    true: Optional[np.ndarray],
) -> pd.DataFrame:
    history = np.asarray(history, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    true = None if true is None else np.asarray(true, dtype=float).reshape(-1)

    n_history = len(history)
    n_future = len(pred)
    total_len = n_history + n_future

    chart = pd.DataFrame(index=np.arange(total_len))
    chart["Observed history"] = np.nan
    chart.loc[: n_history - 1, "Observed history"] = history

    chart["Forecast"] = np.nan
    chart.loc[n_history : total_len - 1, "Forecast"] = pred

    if true is not None and len(true) == n_future:
        chart["Future ground truth"] = np.nan
        chart.loc[n_history : total_len - 1, "Future ground truth"] = true

    chart.index.name = "time_step"
    return chart



# ============================================================
# ONNX inference helpers
# ============================================================


def expected_onnx_path() -> Path:
    """Default ONNX model path used by the Streamlit app."""
    return LOCAL_APP_ONNX_PATH


def find_available_onnx_model(project_dir: Path) -> Optional[Path]:
    """Find a usable ONNX model in the common app/project locations."""
    candidates = [
        LOCAL_APP_ONNX_PATH,
        APP_DIR / "calf_weather_forecast.onnx",
        APP_DIR / "calf_forecast.onnx",
        project_dir / "calf_weather_forecast.onnx",
        project_dir / "calf_forecast.onnx",
        project_dir / "weights" / "calf_weather_forecast.onnx",
        project_dir / "weights" / "calf_forecast.onnx",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def get_onnx_io_info(onnx_path: Path) -> Dict[str, str]:
    """Return lightweight ONNX input/output metadata."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    return {
        "input_name": inputs[0].name,
        "input_shape": str(inputs[0].shape),
        "input_type": str(inputs[0].type),
        "output_name": outputs[0].name,
        "output_shape": str(outputs[0].shape),
        "output_type": str(outputs[0].type),
    }


@st.cache_resource(show_spinner=False)
def load_onnx_session_cached(path_str: str):
    """Cache ONNXRuntime session so repeated predictions are fast."""
    import onnxruntime as ort

    return ort.InferenceSession(path_str, providers=["CPUExecutionProvider"])


def build_onnx_window(
    df: pd.DataFrame,
    cfg: CALFConfig,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], SimpleStandardScaler, List[str], int]:
    """Prepare one normalized input window for ONNXRuntime.

    The exported ONNX model expects the same normalized data used by CALF.
    We reproduce Dataset_Custom-style column order and StandardScaler behavior.
    """
    ordered_df, feature_cols, target_idx = reorder_like_dataset_custom(df, cfg.target)
    values = ordered_df[feature_cols].astype(float).values

    if len(values) < cfg.seq_len:
        raise ValueError(f"Need at least seq_len={cfg.seq_len} rows, but CSV only has {len(values)} rows.")

    scaler = SimpleStandardScaler()
    num_train = max(1, int(len(values) * 0.7))
    scaler.fit(values[:num_train])
    values_scaled = (values - scaler.mean_) / scaler.scale_

    if mode == "Backtest last window with ground truth":
        if len(values_scaled) < cfg.seq_len + cfg.pred_len:
            raise ValueError(
                f"Backtest mode needs at least seq_len + pred_len = {cfg.seq_len + cfg.pred_len} rows, "
                f"but CSV only has {len(values_scaled)} rows. Use future-forecast mode instead."
            )
        x_scaled = values_scaled[-(cfg.seq_len + cfg.pred_len): -cfg.pred_len]
        history_raw = values[-(cfg.seq_len + cfg.pred_len): -cfg.pred_len, target_idx]
        true_raw = values[-cfg.pred_len:, target_idx]
    else:
        x_scaled = values_scaled[-cfg.seq_len:]
        history_raw = values[-cfg.seq_len:, target_idx]
        true_raw = None

    x = x_scaled[np.newaxis, :, :].astype(np.float32)
    return x, history_raw.astype(float), true_raw, scaler, feature_cols, target_idx


def run_onnx_forecast(
    onnx_path: Path,
    df: pd.DataFrame,
    cfg: CALFConfig,
    mode: str,
    save_result: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, float], Path]:
    """Run ONNX inference and optionally save pred/true/input/metrics npy files."""
    x, history_raw, true_raw, scaler, feature_cols, target_idx = build_onnx_window(df, cfg, mode)

    session = load_onnx_session_cached(str(onnx_path.resolve()))
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    expected_features = input_meta.shape[-1]
    if isinstance(expected_features, int) and expected_features != x.shape[-1]:
        raise ValueError(
            f"ONNX model expects {expected_features} input features, but current CSV has {x.shape[-1]}. "
            "Use the same Weather CSV/features used when exporting ONNX."
        )

    pred_scaled = session.run([output_meta.name], {input_meta.name: x})[0]
    pred_target_scaled = extract_series(pred_scaled, 0, target_idx)

    # Convert normalized target prediction back to original Weather scale.
    pred_raw = inverse_target_values(
        pred_target_scaled,
        scaler=scaler,
        n_features=len(feature_cols),
        target_idx=target_idx,
    )

    metrics: Dict[str, float] = {}
    if true_raw is not None and len(true_raw) == len(pred_raw):
        metrics = calculate_basic_metrics(true_raw, pred_raw)

    result_dir = expected_result_dir(cfg)
    if save_result:
        result_dir.mkdir(parents=True, exist_ok=True)
        np.save(result_dir / "input.npy", x)
        np.save(result_dir / "pred.npy", pred_scaled)
        if true_raw is not None:
            # Save a CALF-like true.npy so the existing saved-results tab can read it.
            # Only the target channel is needed for plotting; other channels are zero.
            true_scaled = np.zeros((1, len(true_raw), len(feature_cols)), dtype=np.float32)
            true_scaled[0, :, target_idx] = (true_raw - scaler.mean_[target_idx]) / scaler.scale_[target_idx]
            np.save(result_dir / "true.npy", true_scaled)
            np.save(result_dir / "true_target_raw.npy", true_raw)
        if metrics:
            np.save(result_dir / "metrics.npy", np.array([metrics.get(k, np.nan) for k in METRIC_NAMES]))

    return history_raw, pred_raw, true_raw, metrics, result_dir

# ============================================================
# Placeholder forecasting
# ============================================================


def create_synthetic_weather(n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    date = pd.date_range("2020-01-01", periods=n, freq="h")
    t = np.arange(n)

    # 21 columns to match your Weather experiment shape.
    data = {}
    for i in range(20):
        data[f"feature_{i+1:02d}"] = (
            np.sin(2 * np.pi * t / (24 * (i % 5 + 1)))
            + 0.02 * i
            + rng.normal(0, 0.05, size=n)
        )

    data["OT"] = (
        0.5 * np.sin(2 * np.pi * t / 24)
        + 0.2 * np.sin(2 * np.pi * t / (24 * 7))
        + rng.normal(0, 0.05, size=n)
    )

    return pd.DataFrame({"date": date, **data})


def placeholder_forecast(
    df: pd.DataFrame,
    target: str,
    seq_len: int,
    pred_len: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    y = df[target].astype(float).values

    if len(y) >= seq_len + pred_len:
        history = y[-(seq_len + pred_len): -pred_len]
        true = y[-pred_len:]
    else:
        history = y[-seq_len:]
        true = None

    last = float(history[-1])
    if len(history) >= 8:
        local_slope = (float(history[-1]) - float(history[-8])) / 8.0
    else:
        local_slope = 0.0

    steps = np.arange(1, pred_len + 1)
    seasonal_amp = 0.05 * (np.std(history) + 1e-8)
    pred = last + local_slope * steps + seasonal_amp * np.sin(2 * np.pi * steps / max(24, pred_len))

    metrics = calculate_basic_metrics(true, pred) if true is not None else {}
    return history, pred, true, metrics


# ============================================================
# Sidebar UI
# ============================================================


def sidebar_config(df: pd.DataFrame, project_dir: Path) -> CALFConfig:
    default_target, default_enc_in, feature_cols = infer_target_and_dim(df, "OT")

    st.sidebar.header("Model configuration")

    target = st.sidebar.selectbox(
        "Target column",
        options=feature_cols,
        index=feature_cols.index(default_target),
        help="For your Weather run, this is usually OT.",
    )

    seq_len = st.sidebar.number_input("Input sequence length", min_value=1, value=96, step=1)
    pred_len = st.sidebar.number_input("Prediction length", min_value=1, value=96, step=1)
    label_len = st.sidebar.number_input("Label length", min_value=0, value=0, step=1)

    st.sidebar.divider()
    st.sidebar.subheader("CALF architecture")

    d_model = st.sidebar.number_input("d_model", min_value=1, value=768, step=1)
    n_heads = st.sidebar.number_input("n_heads", min_value=1, value=4, step=1)
    e_layers = st.sidebar.number_input("e_layers", min_value=1, value=2, step=1)
    d_layers = st.sidebar.number_input("d_layers", min_value=1, value=1, step=1)
    d_ff = st.sidebar.number_input("d_ff", min_value=1, value=768, step=1)
    factor = st.sidebar.number_input("factor", min_value=1, value=1, step=1)
    gpt_layers = st.sidebar.number_input("GPT layers", min_value=1, value=6, step=1)

    st.sidebar.divider()
    st.sidebar.subheader("Runtime")

    batch_size = st.sidebar.number_input("Batch size", min_value=1, value=8, step=1)
    num_workers = st.sidebar.number_input("Num workers", min_value=0, value=0, step=1)
    des = st.sidebar.text_input("Experiment description", value="no_drive_corrected")

    enc_in = len([c for c in df.columns if c != "date"])
    model_id = f"weather_CALF_{int(seq_len)}_{int(pred_len)}"

    return CALFConfig(
        project_dir=project_dir,
        root_path="./datasets/weather/",
        data_path="weather.csv",
        model_id=model_id,
        des=des,
        target=target,
        features="M",
        freq="h",
        seq_len=int(seq_len),
        label_len=int(label_len),
        pred_len=int(pred_len),
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
        d_model=int(d_model),
        n_heads=int(n_heads),
        e_layers=int(e_layers),
        d_layers=int(d_layers),
        d_ff=int(d_ff),
        factor=int(factor),
        embed="timeF",
        distil=True,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        gpt_layers=int(gpt_layers),
        word_embedding_path="wte_pca_500.pt",
    )


# ============================================================
# Main app
# ============================================================


def main() -> None:
    show_header()

    st.sidebar.header("Project and data")
    project_dir_input = st.sidebar.text_input("CALF project folder", value=DEFAULT_PROJECT_DIR)
    project_dir = Path(project_dir_input).expanduser().resolve()

    valid_project, missing_project_files = project_is_valid(project_dir)

    if valid_project:
        st.sidebar.success("CALF project detected")
    else:
        st.sidebar.warning("CALF project not fully detected")

    data_path = project_dir / DEFAULT_DATA_RELATIVE
    uploaded_csv = st.sidebar.file_uploader("Upload weather.csv", type=["csv"])

    if uploaded_csv is not None:
        write_uploaded_file(uploaded_csv, data_path)
        st.sidebar.success(f"Uploaded CSV saved to {data_path}")

    if data_path.exists():
        try:
            df = read_weather_csv(data_path)
            data_source = str(data_path)
        except Exception as exc:
            st.error(f"Could not read Weather CSV: {exc}")
            st.stop()
    else:
        df = create_synthetic_weather()
        data_source = "Synthetic placeholder data"
        st.warning(
            "No Weather CSV was found. The UI is using synthetic placeholder data. "
            "Real CALF inference requires your trained Weather CSV at datasets/weather/weather.csv."
        )

    cfg = sidebar_config(df, project_dir)
    expected_ckpt = expected_checkpoint_path(cfg)
    expected_result = expected_result_dir(cfg)

    with st.expander("Detected configuration", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Variables", cfg.enc_in)
        c3.metric("Target", cfg.target)
        c4.metric("Horizon", cfg.pred_len)

        st.write("**Data source:**", data_source)
        st.write("**Expected setting:**")
        st.code(build_setting(cfg), language="text")
        st.write("**Expected checkpoint path:**")
        st.code(str(expected_ckpt), language="text")

    tab_import, tab_demo, tab_onnx, tab_real, tab_results, tab_data = st.tabs(
        [
            "Import required files",
            "Demo placeholder",
            "ONNX inference",
            "Legacy PyTorch CALF",
            "Load saved results",
            "Data preview",
        ]
    )

    # --------------------------------------------------------
    # Required file import mode
    # --------------------------------------------------------
    with tab_import:
        st.subheader("Import all required files")
        st.info(
            "Upload the files individually, or upload one zip file that contains them. "
            "The app will copy each recognized file to the correct folder automatically."
        )

        st.markdown(
            """
            **Recognized filenames**

            - `weather.csv` → saved to `datasets/weather/weather.csv`
            - `calf_weather_forecast.onnx`, `calf_forecast.onnx`, `model.onnx`, or any `.onnx`/`.onxr` → saved to `weights/calf_weather_forecast.onnx`
            - `weather.csv` → saved to `datasets/weather/weather.csv`
            - `wte_pca_500.pt` → saved to the CALF project root, only for legacy PyTorch mode
            - `checkpoint.pth` or `checkpoint.pt` → saved to the expected CALF checkpoint folder, only for legacy PyTorch mode
            - `pred.npy`, `true.npy`, `input.npy`, `metrics.npy` → saved to the expected result folder
            - `.zip` → extracted and recognized files are imported automatically
            """
        )

        st.divider()
        st.markdown("### Download from Google Drive or direct URL")
        st.caption(
            "Use this when Streamlit still shows 200MB upload limit. "
            "The file is downloaded by the server, so it does not go through `st.file_uploader`. "
            "For Google Drive, set sharing to **Anyone with the link can view**."
        )
        st.warning(
            "For Google Drive links, choose the expected filename below, for example `checkpoint.pth`. "
            "Auto-detect can fail for large Drive files because Google may hide the filename behind a confirmation page."
        )

        url_text = st.text_area(
            "Google Drive / direct download URL",
            placeholder="Paste one URL per line, for example: https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing",
            key="url_import_text",
        )

        url_import_type = st.selectbox(
            "Save downloaded file as",
            [
                "Auto detect from downloaded filename",
                "calf_weather_forecast.onnx",
                "weather.csv",
                "wte_pca_500.pt",
                "checkpoint.pth",
                "pred.npy",
                "true.npy",
                "input.npy",
                "metrics.npy",
                "archive.zip",
            ],
            help="Choose the expected filename if the Google Drive link does not expose the real filename.",
        )

        url_filename_override = st.text_input(
            "Optional custom filename override",
            value="",
            help="Usually leave empty. Use this if Auto detect gives a wrong name.",
        )

        download_import_button = st.button("Download URL and import", type="secondary")

        if download_import_button:
            urls = [line.strip() for line in url_text.splitlines() if line.strip()]
            if not urls:
                st.warning("Please paste at least one Google Drive or direct download URL.")
            else:
                import_rows: List[Tuple[str, str, str]] = []
                download_dir = cfg.project_dir / "streamlit_url_downloads"

                for idx, url in enumerate(urls, start=1):
                    try:
                        canonical_name = "" if url_import_type == "Auto detect from downloaded filename" else url_import_type
                        filename_override = url_filename_override.strip() or canonical_name
                        if len(urls) > 1 and url_filename_override.strip():
                            stem = Path(url_filename_override.strip()).stem
                            suffix = Path(url_filename_override.strip()).suffix
                            filename_override = f"{stem}_{idx}{suffix}"

                        downloaded_path = download_file_from_url(url, download_dir, filename_override=filename_override)
                        st.success(f"Downloaded to: {downloaded_path}")

                        imported = import_downloaded_required_file(
                            downloaded_path,
                            cfg,
                            import_filename=canonical_name or downloaded_path.name,
                        )
                        import_rows.extend(imported)
                    except Exception as exc:
                        import_rows.append((url, "Error", str(exc)))

                imported_df = pd.DataFrame(import_rows, columns=["downloaded_file_or_url", "status", "destination_or_message"])
                st.dataframe(imported_df, use_container_width=True)

                if any(row[1] == "Imported" and row[0].lower().endswith(".npy") for row in import_rows):
                    st.session_state["last_result_dir"] = str(expected_result_dir(cfg))

                if any(row[1] == "Imported" for row in import_rows):
                    st.success("URL import finished. Check the required-file status below.")

        st.divider()

        uploaded_required_files = st.file_uploader(
            "Upload required files or one zip archive",
            type=["csv", "pt", "pth", "npy", "onnx", "onxr", "zip"],
            accept_multiple_files=True,
            key="required_files_uploader",
        )

        import_button = st.button("Import uploaded files", type="primary")

        if import_button:
            if not uploaded_required_files:
                st.warning("Please choose at least one file first.")
            else:
                import_rows: List[Tuple[str, str, str]] = []
                for uploaded_file in uploaded_required_files:
                    if uploaded_file.name.lower().endswith(".zip"):
                        import_rows.extend(import_required_zip(uploaded_file, cfg))
                    else:
                        try:
                            import_rows.append(save_uploaded_required_file(uploaded_file, cfg))
                        except Exception as exc:
                            import_rows.append((uploaded_file.name, "Error", str(exc)))

                imported_df = pd.DataFrame(import_rows, columns=["uploaded_file", "status", "destination_or_message"])
                st.dataframe(imported_df, use_container_width=True)

                if any(row[1] == "Imported" and row[0].lower().endswith(".npy") for row in import_rows):
                    st.session_state["last_result_dir"] = str(expected_result_dir(cfg))

                if any(row[1] == "Imported" for row in import_rows):
                    st.success("Import finished. Open the ONNX inference tab to run `.onnx`, or the saved-results tab to visualize `.npy` outputs.")

        st.write("**Current required-file status:**")
        status_df = required_file_status(cfg)
        st.dataframe(status_df, use_container_width=True)

    # --------------------------------------------------------
    # Demo mode
    # --------------------------------------------------------
    with tab_demo:
        st.subheader("Demo placeholder forecast")
        st.info(
            "This mode is only for UI testing. It does not use CALF weights. "
            "Use the ONNX inference tab after importing `calf_weather_forecast.onnx`."
        )

        if st.button("Run placeholder forecast", type="primary"):
            history, pred, true, metrics = placeholder_forecast(
                df=df,
                target=cfg.target,
                seq_len=cfg.seq_len,
                pred_len=cfg.pred_len,
            )
            if metrics:
                display_metrics(metrics)
            else:
                st.caption("Ground truth is unavailable because the data is shorter than seq_len + pred_len.")

            chart_df = make_forecast_chart_data(history=history, pred=pred, true=true)
            st.line_chart(chart_df, use_container_width=True)

    # --------------------------------------------------------
    # ONNX inference mode
    # --------------------------------------------------------
    with tab_onnx:
        st.subheader("ONNX inference without torch")
        st.success(
            "Use this tab with your exported `calf_weather_forecast.onnx`. "
            "It runs with `onnxruntime` only, so Streamlit Cloud does not need torch, transformers, or peft."
        )

        current_onnx = find_available_onnx_model(project_dir)
        default_onnx_text = str(current_onnx if current_onnx is not None else expected_onnx_path())

        onnx_path_text = st.text_input(
            "ONNX model path",
            value=default_onnx_text,
            help="Upload/import your ONNX file first, or paste the path to an existing .onnx file.",
        )
        onnx_path = Path(onnx_path_text).expanduser().resolve()

        c1, c2, c3 = st.columns(3)
        c1.metric("ONNX model", "OK" if onnx_path.exists() else "Missing")
        c2.metric("Weather CSV", "OK" if data_path.exists() else "Synthetic")
        c3.metric("Input features", cfg.enc_in)

        if onnx_path.exists():
            try:
                info = get_onnx_io_info(onnx_path)
                with st.expander("ONNX model info", expanded=False):
                    st.json(info)
            except Exception as exc:
                st.warning(f"Could not read ONNX metadata yet: {exc}")

        forecast_mode = st.radio(
            "Forecast mode",
            ["Backtest last window with ground truth", "Forecast after latest row"],
            index=0,
            help=(
                "Backtest uses the final seq_len + pred_len rows, so the app can compare prediction with existing ground truth. "
                "Forecast-after-latest uses the final seq_len rows and predicts future values without ground truth."
            ),
        )

        save_onnx_results = st.checkbox(
            "Save ONNX output arrays to the expected result folder",
            value=True,
        )

        if st.button("Run ONNX Weather inference", type="primary"):
            try:
                if not onnx_path.exists():
                    raise FileNotFoundError(
                        f"ONNX model not found: {onnx_path}. Import `calf_weather_forecast.onnx` first."
                    )

                history, pred, true, metrics, result_dir = run_onnx_forecast(
                    onnx_path=onnx_path,
                    df=df,
                    cfg=cfg,
                    mode=forecast_mode,
                    save_result=save_onnx_results,
                )

                st.session_state["last_result_dir"] = str(result_dir)

                if metrics:
                    st.write("**Backtest metrics on target column:**")
                    display_metrics(metrics)
                else:
                    st.info("Ground truth is not available in future-forecast mode, so metrics are not computed.")

                chart_df = make_forecast_chart_data(history=history, pred=pred, true=true)
                st.line_chart(chart_df, use_container_width=True)

                out_df = pd.DataFrame({
                    "step": np.arange(1, len(pred) + 1),
                    "forecast": pred,
                })
                if true is not None and len(true) == len(pred):
                    out_df["ground_truth"] = true
                    out_df["absolute_error"] = np.abs(true - pred)

                st.write("**Forecast table:**")
                st.dataframe(out_df, use_container_width=True)

                st.download_button(
                    "Download ONNX forecast CSV",
                    data=out_df.to_csv(index=False).encode("utf-8"),
                    file_name="calf_weather_onnx_forecast.csv",
                    mime="text/csv",
                )

                if save_onnx_results:
                    st.success(f"Saved ONNX output files to: {result_dir}")

            except Exception as exc:
                st.exception(exc)

    # --------------------------------------------------------
    # Real CALF inference mode
    # --------------------------------------------------------
    with tab_real:
        st.subheader("Legacy PyTorch CALF checkpoint inference")
        st.warning(
            "This lightweight Streamlit Cloud build does not install torch/transformers/peft by default. "
            "Use this tab only in a local/Colab environment where CALF dependencies are installed. "
            "For Streamlit Cloud, run CALF elsewhere and upload/load saved result arrays here."
        )

        if not valid_project:
            st.error("CALF project files are missing. Real inference cannot run yet.")
            st.write("Missing files:")
            st.code("\n".join(missing_project_files), language="text")
            st.markdown(
                "Place this `app.py` beside your CALF project or set `CALF_PROJECT_DIR` / sidebar path correctly."
            )
        else:
            uploaded_ckpt = st.file_uploader(
                "Upload trained checkpoint.pth",
                type=["pth", "pt"],
                help="Upload the Weather checkpoint produced after training. The app will copy it to the expected CALF checkpoint folder.",
            )

            manual_ckpt_path = st.text_input(
                "Or enter existing checkpoint path",
                value=str(LOCAL_APP_WEIGHT_PATH if LOCAL_APP_WEIGHT_PATH.exists() else project_dir / DEFAULT_WEIGHT_RELATIVE_PLACEHOLDER),
            )

            apply_patches = st.checkbox("Apply runtime compatibility patches", value=True)
            copy_to_expected = st.checkbox("Copy selected checkpoint to expected CALF folder", value=True)

            selected_ckpt_path: Optional[Path] = None

            if uploaded_ckpt is not None:
                tmp_upload_dir = project_dir / "streamlit_uploaded_weights"
                tmp_upload_dir.mkdir(parents=True, exist_ok=True)
                uploaded_path = tmp_upload_dir / "checkpoint.pth"
                write_uploaded_file(uploaded_ckpt, uploaded_path)
                selected_ckpt_path = uploaded_path
                st.success(f"Uploaded checkpoint saved to {uploaded_path}")
            elif manual_ckpt_path.strip():
                selected_ckpt_path = Path(manual_ckpt_path).expanduser().resolve()

            status_cols = st.columns(3)
            status_cols[0].metric("Project", "OK" if valid_project else "Missing")
            status_cols[1].metric("wte_pca_500.pt", "OK" if (project_dir / "wte_pca_500.pt").exists() else "Missing")
            status_cols[2].metric("Expected checkpoint", "OK" if expected_ckpt.exists() else "Missing")

            st.write("**Inference command preview:**")
            cmd_preview = build_calf_command(cfg, is_training=0)
            st.code(" ".join(shlex.quote(x) for x in cmd_preview), language="bash")

            run_button = st.button("Run real CALF inference", type="primary")

            if run_button:
                try:
                    if apply_patches:
                        with st.spinner("Applying runtime patches..."):
                            patch_messages = apply_runtime_patches(project_dir)
                        with st.expander("Patch log", expanded=False):
                            for msg in patch_messages:
                                st.write("-", msg)

                    if not (project_dir / "wte_pca_500.pt").exists():
                        raise FileNotFoundError(
                            "Missing wte_pca_500.pt. Run pca.py in CALF first, or copy wte_pca_500.pt into the CALF folder."
                        )

                    if selected_ckpt_path is not None and selected_ckpt_path.exists() and copy_to_expected:
                        copy_checkpoint_to_expected(selected_ckpt_path, expected_ckpt)
                        st.success(f"Checkpoint copied to expected path: {expected_ckpt}")

                    if not expected_ckpt.exists():
                        raise FileNotFoundError(
                            "No checkpoint found at the expected CALF folder. "
                            "Upload checkpoint.pth or correct the checkpoint path."
                        )

                    # Ensure the current df is saved to the path CALF will read.
                    data_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(data_path, index=False)

                    cmd = build_calf_command(cfg, is_training=0)
                    rc = run_subprocess_with_streamlit_logs(cmd, cwd=project_dir)
                    if rc != 0:
                        st.stop()

                    result_dir = expected_result if expected_result.exists() else latest_result_dir(project_dir, cfg.model_id)
                    if result_dir is None:
                        raise FileNotFoundError("No result folder found after inference.")

                    st.session_state["last_result_dir"] = str(result_dir)
                    st.success(f"Result folder: {result_dir}")

                except Exception as exc:
                    st.exception(exc)

    # --------------------------------------------------------
    # Saved results mode
    # --------------------------------------------------------
    with tab_results:
        st.subheader("Load and visualize saved CALF results")

        default_result_dir = st.session_state.get("last_result_dir", str(expected_result))
        result_dir_text = st.text_input("Result folder", value=default_result_dir)
        result_dir = Path(result_dir_text).expanduser().resolve()

        use_inverse_scaling = st.checkbox(
            "Apply Weather StandardScaler inverse transform for plot",
            value=True,
            help="Use this if pred.npy/true.npy/input.npy are still in normalized CALF space. Disable if your arrays are already inverse-transformed.",
        )

        if st.button("Load result arrays"):
            try:
                arrays = load_result_arrays(result_dir)
                if not arrays:
                    raise FileNotFoundError(f"No .npy result arrays found in {result_dir}")

                st.write("**Loaded files:**", ", ".join([f"{k}.npy" for k in arrays.keys()]))

                if "metrics" in arrays:
                    metrics_arr = arrays["metrics"].reshape(-1)
                    metrics = {
                        name: float(metrics_arr[i])
                        for i, name in enumerate(METRIC_NAMES)
                        if i < len(metrics_arr)
                    }
                    display_metrics(metrics)

                if "pred" not in arrays or "true" not in arrays:
                    raise FileNotFoundError("pred.npy and true.npy are required for visualization.")

                pred_arr = arrays["pred"]
                true_arr = arrays["true"]
                input_arr = arrays.get("input")

                st.write("**Array shapes:**")
                shape_info = {k: str(v.shape) for k, v in arrays.items()}
                st.json(shape_info)

                n_samples = int(pred_arr.shape[0]) if pred_arr.ndim == 3 else 1

                # Streamlit slider requires min_value < max_value.
                # For a single saved sample, there is nothing to select.
                if n_samples <= 1:
                    sample_idx = 0
                    st.info("Only one sample is available, so sample index is fixed to 0.")
                else:
                    sample_idx = st.slider("Sample index", 0, n_samples - 1, 0)

                scaler, feature_cols, target_idx = fit_train_scaler(df, cfg.target)
                n_features = len(feature_cols)

                pred = extract_series(pred_arr, sample_idx, target_idx)
                true = extract_series(true_arr, sample_idx, target_idx)

                if input_arr is not None:
                    history = extract_series(input_arr, sample_idx, target_idx)
                else:
                    # Fallback if input.npy was not saved by older code.
                    y = df[cfg.target].astype(float).values
                    history = y[-cfg.seq_len:]

                if use_inverse_scaling:
                    history = inverse_target_values(history, scaler, n_features, target_idx)
                    pred = inverse_target_values(pred, scaler, n_features, target_idx)
                    true = inverse_target_values(true, scaler, n_features, target_idx)

                local_metrics = calculate_basic_metrics(true, pred)
                st.write("**Target-only metrics for selected sample:**")
                display_metrics(local_metrics)

                chart_df = make_forecast_chart_data(history=history, pred=pred, true=true)
                st.line_chart(chart_df, use_container_width=True)

                out_df = pd.DataFrame({
                    "step": np.arange(1, len(pred) + 1),
                    "ground_truth": true,
                    "forecast": pred,
                    "absolute_error": np.abs(true - pred),
                })
                st.dataframe(out_df, use_container_width=True)

                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download selected forecast CSV",
                    data=csv_bytes,
                    file_name="calf_weather_selected_forecast.csv",
                    mime="text/csv",
                )

            except Exception as exc:
                st.exception(exc)

    # --------------------------------------------------------
    # Data preview
    # --------------------------------------------------------
    with tab_data:
        st.subheader("Weather data preview")
        st.dataframe(df.head(100), use_container_width=True)
        st.write("**Columns:**")
        st.code("\n".join(df.columns.astype(str)), language="text")

        target_series = df[cfg.target].astype(float).values
        recent = target_series[-min(len(target_series), 500):]
        st.line_chart(pd.DataFrame({cfg.target: recent}), use_container_width=True)


if __name__ == "__main__":
    main()
