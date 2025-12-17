# Image Forensics – Simple Document Processor

A compact tool that detects a document inside a photo, fixes perspective, crops safely, and reports the document’s **rotation angle** in **[0, 360)**. Tuned to behave well on gray/low‑contrast desks and close‑ups, with over‑crop guards. Typical target runtime: ~**100 ms** per image (with `max_dim≈1024`).

---

## Features
- Document quad detection with multiple masks & fallbacks
- Angle of the document’s top edge vs +X in **[0, 360)** (no negatives)
- Perspective warp → cautious white trim → landscape normalize
- Context padding so results never feel cramped
- Returns:
  - NumPy ndarray (BGR) for library use, or
  - **Base64 PNG** for CLI/API

Optional assists (auto‑off if not installed): **pytesseract** (OCR hull), **ultralytics YOLOv8** (candidates).

---

## Install (Poetry)
```bash
poetry install
# if pyproject changed:
poetry lock && poetry install
```

Sanity check:
```bash
poetry run python -c "import cv2, numpy as np; print('OpenCV OK')"
```

---

## Library Usage

### From a file path
```python
from image_forensics.pipeline import process_image_path, to_base64_png

angle, crop, ms = process_image_path("data/raw/sample.jpg", max_dim=1024)
print(f"angle={angle:.1f}°, time={ms:.1f} ms, shape={crop.shape}")

b64_png = to_base64_png(crop)  # optional
```

### From an ndarray
```python
import cv2 as cv
from image_forensics.pipeline import process_image_array

img = cv.imread("data/raw/sample.jpg", cv.IMREAD_COLOR)
angle, crop, ms = process_image_array(img, max_dim=1024)
```

**Returns:** `angle` (float `[0,360)`), `crop` (np.ndarray BGR), `ms` (elapsed ms).

---

## CLI (Batch)
Module form (works without console script):
```bash
poetry run python -m image_forensics.cli   --inp data/raw   --out outputs   --max-dim 1024   --threads 2
```

If you added a console script `image-forensics` in `pyproject.toml`:
```bash
poetry run image-forensics --inp data/raw --out outputs --max-dim 1024 --threads 2
```

Example output:
```
invoice1.jpg                  angle= 12.4°  time= 95.7 ms  -> invoice1_crop.png
passport_desk.png             angle=178.9°  time=101.3 ms  -> passport_desk_crop.png

AVG 98.6 ms | P95 122.4 ms | n=42
```

---

## HTTP API (Optional)
Start:
```bash
poetry run uvicorn image_forensics.api:app --host 127.0.0.1 --port 8000
```

**POST** `/process` (multipart `file`) → JSON:
```json
{
  "angle": 12.4,
  "elapsed_ms": 96.2,
  "image_b64": "<PNG base64>"
}
```

---

## Behavior & Guards
- **Close‑up safety:** conservative trim with auto‑revert if >10% area would be removed.
- **Gray/low‑contrast robustness:** LAB normalization + CLAHE + relative “whiteness” mask.
- **Touching‑frame logic:** trims less and adds more context when the quad hugs borders.
- **Angle:** measured from the document’s *top edge* against +X; normalized to `[0, 360)`.

---

## Configuration
- `max_dim` (default `1024`) controls downscale for detection (speed/quality).
- Trim thresholds are guarded (auto‑revert if too aggressive).
- Optional assists:
  - OCR: install `pytesseract` and system Tesseract
  - YOLO: `pip install ultralytics` and have a YOLOv8 model available

---

## Project Structure
```
image_forensics/
  ├─ image_forensics/
  │   ├─ __init__.py
  │   ├─ core.py           # detection, masks, refinement, helpers
  │   ├─ pipeline.py       # process_image_* functions (angle + crop)
  │   ├─ cli.py            # batch CLI (folder in → outputs)
  │   └─ api.py            # FastAPI app (optional)
  ├─ data/                 # (ignored) sample inputs
  ├─ outputs/              # (ignored) results
  ├─ pyproject.toml
  └─ README.md
```

---

## Troubleshooting
- **WSL uvicorn “python not found”**
  ```bash
  rm -rf .venv
  poetry env use python3
  poetry install
  ```
- **“Command not found: image-forensics”**  
  Use `python -m image_forensics.cli` or ensure `[project.scripts]` is set and reinstall.
- **Over‑cropping on some sets**  
  Reduce trim aggressiveness or rely on the built‑in `should_trim_white(...)` gate.
- **Slow on very large photos**  
  Lower `--max-dim` to `960` or `896`.

---

## License
MIT

## Acknowledgements
OpenCV + NumPy. Optional: Tesseract OCR, Ultralytics YOLO.
