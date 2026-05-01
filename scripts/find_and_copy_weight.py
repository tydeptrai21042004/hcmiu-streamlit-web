# Run this in Colab after CALF training finishes.
# It finds the latest trained checkpoint and copies it to /content/weather_calf_checkpoint.pth

from pathlib import Path
import shutil

ckpt_root = Path("/content/CALF/checkpoints")
ckpts = sorted(
    ckpt_root.rglob("checkpoint.pth"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if not ckpts:
    raise FileNotFoundError("No checkpoint.pth found. Run training first.")

latest = ckpts[0]
out = Path("/content/weather_calf_checkpoint.pth")
shutil.copy2(latest, out)

print("[OK] Latest checkpoint:")
print(latest)
print("[OK] Copied to:")
print(out)
