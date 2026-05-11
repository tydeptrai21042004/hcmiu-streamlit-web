# Google Drive / URL import fix

This version adds a server-side downloader inside the **Import required files** tab.

Use it when Streamlit still displays `200MB per file` in `st.file_uploader`.
The downloader does not use `st.file_uploader`; it downloads the file from a public Google Drive or direct-download URL directly to the server disk.

## How to use

1. Upload your large file to Google Drive.
2. Right click the file -> Share.
3. Set access to **Anyone with the link can view**.
4. Copy the link.
5. Open the Streamlit app.
6. Go to **Import required files**.
7. Paste the link in **Google Drive / direct download URL**.
8. Choose the expected filename in **Save downloaded file as**:
   - `weather.csv`
   - `wte_pca_500.pt`
   - `checkpoint.pth`
   - `pred.npy`, `true.npy`, `input.npy`, or `metrics.npy`
   - `archive.zip`
9. Click **Download URL and import**.

The app will save the file to the same expected destination used by the upload importer.

## Notes

- Private Google Drive files will fail. Use **Anyone with the link can view**.
- Streamlit Cloud storage/RAM may still be limited. A 1GB model is safer on a local machine, VPS, or Hugging Face Space with enough disk/RAM.
- The normal file upload widget may still show `200MB per file`; ignore it and use the URL downloader section.
