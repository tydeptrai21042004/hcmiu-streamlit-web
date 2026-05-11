# CALF Weather Streamlit App — Deploy Ready

This folder contains a Streamlit UI for your corrected CALF Weather forecasting notebook.

The app supports:

1. **Demo placeholder mode** — works immediately without trained weights.
2. **Real CALF inference mode** — uses your trained `checkpoint.pth`.
3. **Saved results mode** — visualizes `input.npy`, `pred.npy`, `true.npy`, and `metrics.npy`.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```


---

## 1 GB upload support

This package includes `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 1024
```

This allows `st.file_uploader` to accept files up to about 1 GB when the server has enough RAM and disk space. For very large `checkpoint.pth` files, the recommended local method is still to copy the file directly into `weights/` or into the expected CALF checkpoint folder, then enter the path in the app instead of uploading through the browser.

---

## Important folder structure

```text
calf_weather_streamlit_deploy/
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── data/
│   ├── weather_sample.csv
│   └── README.md
├── weights/
│   ├── PUT_CHECKPOINT_HERE.txt
│   └── README.md
├── results/
│   └── README.md
└── scripts/
    └── find_and_copy_weight.py
```

---

## Where to put your trained weight

After training in Colab, your checkpoint is usually:

```text
/content/CALF/checkpoints/long_term_forecast_weather_CALF_96_96_CALF_custom_ftM_sl96_ll0_pl96_dm768_nh4_el2_dl1_df768_fc1_ebtimeF_dtFalse_no_drive_corrected_gpt6_0/checkpoint.pth
```

Copy it to your deploy folder, or upload it in the Streamlit sidebar.

Recommended deploy-folder location:

```text
weights/weather_calf_checkpoint.pth
```

Inside the app, for real CALF inference, you can upload this file or enter the existing checkpoint path.

---

## Required CALF files for real inference

A `checkpoint.pth` file is not enough by itself. It contains only trained weights.

For real inference, the app also needs the CALF source folder with:

```text
CALF/
├── run.py
├── models/
│   ├── CALF.py
│   └── GPT2_arch.py
├── exp/
│   └── exp_long_term_forecasting.py
├── datasets/
│   └── weather/
│       └── weather.csv
└── wte_pca_500.pt
```

In the app sidebar, set **CALF project folder** to your CALF folder.

For Colab, this is usually:

```text
/content/CALF
```

For local deployment, place your CALF repo beside the app or set the path manually.

---

## Weather CSV

Put your real Weather dataset here:

```text
data/weather.csv
```

or upload it from the sidebar.

If no real CSV is found, the app uses `data/weather_sample.csv` only for UI testing.

---

## Saved results mode

You can also copy result folders from your notebook:

```text
/content/CALF/results/<result_folder>/
```

A valid result folder contains:

```text
input.npy
pred.npy
true.npy
metrics.npy
```

Then load the folder in the **Load saved results** tab.

---

## Streamlit Cloud deployment

1. Upload this whole folder to a GitHub repository.
2. Make sure `app.py` is at the repository root.
3. Deploy with Streamlit Cloud.
4. Add your trained checkpoint and `wte_pca_500.pt` by either:
   - committing them if file size is acceptable, or
   - uploading them in the app sidebar after deployment.

---

## Notes

- The **Demo placeholder** mode is not CALF. It is only for UI testing.
- The **Real CALF inference** mode runs the actual CALF code through `run.py --is_training 0`.
- The default checkpoint folder name has been corrected to match your notebook setting with `dtFalse` and `no_drive_corrected_gpt6`.
