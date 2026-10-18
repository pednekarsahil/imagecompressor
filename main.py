"""
Image Compressor 2026 — single-file web app
Run:  python app.py
Open: http://127.0.0.1:8000
"""

import os
import io
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

# ══════════════════════════════════════════════════════════════
#  PRESETS & CORE COMPRESSION (identical to your Tkinter version)
# ══════════════════════════════════════════════════════════════
PRESETS = {
    "lossless":   {"quality": 100, "subsampling": 0, "optimize": True},
    "high":       {"quality": 90,  "subsampling": 0, "optimize": True},
    "balanced":   {"quality": 78,  "subsampling": 2, "optimize": True},
    "web":        {"quality": 65,  "subsampling": 2, "optimize": True},
    "thumbnail":  {"quality": 50,  "subsampling": 2, "optimize": True, "max_dimension": 600},
    "aggressive": {"quality": 35,  "subsampling": 2, "optimize": True},
}

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def _prepare_image(img, max_dimension=None, keep_exif=True, auto_orient=True):
    if auto_orient:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
    exif_bytes = b""
    if keep_exif and "exif" in img.info:
        exif_bytes = img.info["exif"]
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        conv = img.convert("RGBA") if img.mode == "P" else img
        bg.paste(conv, mask=conv.split()[-1])
        img = bg
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max_dimension:
        w, h = img.size
        longest = max(w, h)
        if longest > max_dimension:
            r = max_dimension / longest
            img = img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.LANCZOS)
    return img, exif_bytes


def _save_kwargs(fmt, quality, subsampling, optimize, progressive, exif_bytes):
    kw = {"optimize": optimize}
    if fmt in ("JPEG", "WEBP"):
        kw["quality"] = int(quality)
        if fmt == "JPEG":
            kw["subsampling"] = subsampling
            kw["progressive"] = progressive
        if exif_bytes:
            kw["exif"] = exif_bytes
    elif fmt == "PNG":
        kw["compress_level"] = max(0, min(9, int((100 - quality) / 11)))
        if exif_bytes:
            kw["exif"] = exif_bytes
    elif fmt == "TIFF":
        kw["compression"] = "tiff_deflate" if quality < 100 else "tiff_lzw"
    return kw


def _fmt_from_name(name):
    ext = os.path.splitext(name)[1].lower()
    return {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP",
            ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF"}.get(ext, "JPEG")


def _encode(img, fmt, kw):
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def compress_bytes(input_bytes, filename, quality=None, preset=None,
                   max_size_kb=None, max_dimension=None, output_format=None,
                   keep_exif=True):
    cfg = {"quality": 78, "subsampling": 2, "optimize": True, "max_dimension": None}
    if preset and preset in PRESETS:
        cfg.update(PRESETS[preset])
    if quality is not None:       cfg["quality"] = quality
    if max_dimension is not None: cfg["max_dimension"] = max_dimension

    quality, subsampling = cfg["quality"], cfg["subsampling"]
    max_dimension, optimize = cfg["max_dimension"], cfg["optimize"]
    fmt = output_format or _fmt_from_name(filename)

    img = Image.open(io.BytesIO(input_bytes))
    orig_w, orig_h = img.size
    img, exif_bytes = _prepare_image(img, max_dimension, keep_exif)
    save_kw = _save_kwargs(fmt, quality, subsampling, optimize, True, exif_bytes)

    if max_size_kb:
        target = max_size_kb * 1024
        best_bytes, best_q = None, 1
        low, high = 5, 95
        for _ in range(12):
            q = (low + high) // 2
            trial = dict(save_kw)
            if fmt in ("JPEG", "WEBP"):
                trial["quality"] = q
            elif fmt == "PNG":
                trial["compress_level"] = max(0, min(9, int((100 - q) / 11)))
            try:
                data = _encode(img, fmt, trial)
            except Exception:
                break
            if len(data) <= target:
                best_bytes, best_q = data, q
                low = q + 1
            else:
                high = q - 1
            if low > high:
                break
        if best_bytes is None:
            scale = 0.9
            while scale > 0.15:
                new = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                timg = img.resize(new, Image.LANCZOS)
                trial = dict(save_kw)
                if fmt in ("JPEG", "WEBP"):
                    trial["quality"] = 40
                data = _encode(timg, fmt, trial)
                if len(data) <= target:
                    best_bytes, img = data, timg
                    break
                scale -= 0.1
        if best_bytes is None:
            raise RuntimeError("Could not reach target size.")
        out_bytes = best_bytes
    else:
        out_bytes = _encode(img, fmt, save_kw)

    ext_map = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp",
               "BMP": ".bmp", "TIFF": ".tiff"}
    base = os.path.splitext(filename)[0]
    out_name = f"{base}_compressed{ext_map.get(fmt, '.jpg')}"

    return out_bytes, out_name, {
        "original_kb": len(input_bytes) / 1024,
        "compressed_kb": len(out_bytes) / 1024,
        "reduction_pct": (1 - len(out_bytes) / len(input_bytes)) * 100 if input_bytes else 0,
        "original_px": [orig_w, orig_h],
        "new_px": [img.width, img.height],
        "format": fmt,
    }


# ══════════════════════════════════════════════════════════════
#  FASTAPI
# ══════════════════════════════════════════════════════════════
app = FastAPI(title="Image Compressor 2026")
OUTPUT_DIR = os.path.abspath("output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_PAGE


@app.post("/api/compress")
async def api_compress(
    file: UploadFile = File(...),
    preset: str = Form(None),
    quality: int = Form(None),
    max_size_kb: float = Form(None),
    max_dimension: int = Form(None),
    output_format: str = Form(None),
    keep_exif: bool = Form(True),
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    try:
        data = await file.read()
        out_bytes, out_name, stats = compress_bytes(
            data, file.filename,
            quality=quality, preset=preset,
            max_size_kb=max_size_kb, max_dimension=max_dimension,
            output_format=output_format, keep_exif=keep_exif,
        )
    except Exception as e:
        raise HTTPException(500, f"Compression failed: {e}")

    save_name = f"{uuid.uuid4().hex[:8]}_{out_name}"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    with open(save_path, "wb") as f:
        f.write(out_bytes)

    return JSONResponse({
        "ok": True,
        "filename": out_name,
        "download_url": f"/output/{save_name}",
        "stats": stats,
    })


# ══════════════════════════════════════════════════════════════
#  EMBEDDED HTML / CSS / JS — 2026 UI
# ══════════════════════════════════════════════════════════════
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Image Compressor · 2026</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #07090f;
    --bg-2: #0d1220;
    --panel: rgba(20, 28, 46, 0.55);
    --panel-border: rgba(120, 140, 200, 0.14);
    --text: #e8eeff;
    --text-dim: #8b98bd;
    --accent: #6d5efc;
    --accent-2: #22d3ee;
    --accent-3: #f472b6;
    --success: #22c55e;
    --danger: #ef4444;
    --radius: 20px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
    -webkit-font-smoothing: antialiased;
  }

  /* Ambient orbs */
  .orbs { position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none; }
  .orb {
    position: absolute; border-radius: 50%;
    filter: blur(120px); opacity: 0.32;
    animation: drift 22s ease-in-out infinite alternate;
  }
  .orb.a { width: 560px; height: 560px; background: var(--accent);   top: -200px; left: -160px; }
  .orb.b { width: 460px; height: 460px; background: var(--accent-2); bottom: -180px; right: -140px; animation-delay: -8s; }
  .orb.c { width: 420px; height: 420px; background: var(--accent-3); top: 45%; left: 45%; animation-delay: -14s; opacity: 0.22; }
  @keyframes drift {
    from { transform: translate(0, 0) scale(1); }
    to   { transform: translate(70px, -50px) scale(1.18); }
  }

  /* Nav */
  .nav {
    position: sticky; top: 0; z-index: 10;
    display: flex; justify-content: space-between; align-items: center;
    padding: 16px 32px;
    backdrop-filter: blur(22px) saturate(140%);
    background: rgba(7, 9, 15, 0.65);
    border-bottom: 1px solid var(--panel-border);
  }
  .brand { display: flex; align-items: center; gap: 12px; font-weight: 700; font-size: 17px; letter-spacing: -0.3px; }
  .logo {
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: grid; place-items: center;
    font-size: 18px; font-weight: 900;
    box-shadow: 0 6px 22px -6px var(--accent);
  }
  .brand strong {
    background: linear-gradient(90deg, var(--accent-2), var(--accent));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .nav-links { display: flex; align-items: center; gap: 22px; font-size: 14px; color: var(--text-dim); }
  .nav-links a { color: var(--text-dim); text-decoration: none; transition: color .2s; }
  .nav-links a:hover { color: var(--text); }
  .pill {
    font-size: 11px; font-weight: 600; padding: 5px 10px;
    border-radius: 999px; background: rgba(109, 94, 252, 0.14);
    color: #b3aaff; border: 1px solid rgba(109, 94, 252, 0.3);
    font-family: 'JetBrains Mono', monospace;
  }

  /* Container */
  .container {
    position: relative; z-index: 1;
    max-width: 980px; margin: 0 auto;
    padding: 60px 24px 80px;
  }

  /* Hero */
  .hero { text-align: center; margin-bottom: 48px; }
  .hero-badge {
    display: inline-block; font-size: 13px; font-weight: 500;
    padding: 7px 14px; border-radius: 999px;
    background: rgba(34, 211, 238, 0.08);
    border: 1px solid rgba(34, 211, 238, 0.22);
    color: #7de8f5; margin-bottom: 24px;
  }
  h1 {
    font-size: clamp(32px, 5vw, 54px); line-height: 1.05;
    font-weight: 800; letter-spacing: -1.5px; margin-bottom: 18px;
  }
  .grad {
    background: linear-gradient(90deg, #22d3ee, #6d5efc 45%, #f472b6);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .sub { font-size: 17px; color: var(--text-dim); max-width: 640px; margin: 0 auto; line-height: 1.55; }

  /* Dropzone */
  .dropzone {
    border: 2px dashed rgba(120, 140, 200, 0.25);
    border-radius: var(--radius);
    padding: 56px 32px; text-align: center;
    background: rgba(20, 28, 46, 0.3);
    transition: all .25s ease;
    cursor: pointer; margin-bottom: 26px;
    position: relative; overflow: hidden;
  }
  .dropzone:hover, .dropzone.drag {
    border-color: var(--accent);
    background: rgba(109, 94, 252, 0.08);
    box-shadow: 0 0 0 6px rgba(109, 94, 252, 0.08);
  }
  .dz-icon {
    font-size: 42px; margin-bottom: 12px;
    background: linear-gradient(135deg, var(--accent-2), var(--accent));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .dropzone h3 { font-size: 20px; font-weight: 700; margin-bottom: 6px; }
  .dropzone p { color: var(--text-dim); font-size: 14px; }
  .dropzone small { display: block; margin-top: 12px; color: var(--text-dim); font-size: 12px; opacity: 0.7; }
  .link-btn {
    background: none; border: none; color: var(--accent-2);
    font: inherit; cursor: pointer; text-decoration: underline;
    text-underline-offset: 3px;
  }

  /* File list */
  .files-panel {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: var(--radius);
    padding: 18px 20px; margin-bottom: 22px;
    backdrop-filter: blur(20px);
    display: none;
  }
  .files-panel.active { display: block; }
  .files-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px;
  }
  .files-header h3 { font-size: 14px; font-weight: 600; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; }
  .clear-btn {
    background: rgba(239, 68, 68, 0.1); color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.25);
    padding: 6px 12px; border-radius: 8px;
    font-size: 12px; font-weight: 600; cursor: pointer; transition: all .2s;
  }
  .clear-btn:hover { background: rgba(239, 68, 68, 0.2); }
  .file-list { display: flex; flex-direction: column; gap: 6px; max-height: 220px; overflow-y: auto; }
  .file-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 14px; border-radius: 10px;
    background: rgba(13, 18, 32, 0.6);
    font-size: 13px;
  }
  .file-name { color: var(--text); font-family: 'JetBrains Mono', monospace; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
  .file-size { color: var(--text-dim); font-size: 12px; margin-left: 12px; font-variant-numeric: tabular-nums; }

  /* Panel */
  .panel {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: var(--radius);
    padding: 26px;
    backdrop-filter: blur(20px) saturate(140%);
    margin-bottom: 26px;
  }
  .panel-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 22px;
  }
  .panel-header h2 { font-size: 17px; font-weight: 700; letter-spacing: -0.3px; }
  .pill-sm {
    font-size: 11px; font-weight: 600; padding: 4px 10px;
    border-radius: 999px; background: rgba(109, 94, 252, 0.12);
    color: #b3aaff; border: 1px solid rgba(109, 94, 252, 0.28);
  }
  .pill-sm.success { background: rgba(34, 197, 94, 0.1); color: #86efac; border-color: rgba(34, 197, 94, 0.3); }

  /* Setting rows */
  .setting {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 18px; align-items: center;
    padding: 16px 0;
    border-bottom: 1px solid rgba(120, 140, 200, 0.08);
  }
  .setting:last-child { border-bottom: none; }
  .setting-info label {
    display: block; font-size: 14px; font-weight: 600; margin-bottom: 3px;
  }
  .hint { font-size: 12px; color: var(--text-dim); line-height: 1.4; }
  .star { color: #fbbf24; margin-right: 4px; }

  /* Target size highlight */
  .setting.target {
    background: linear-gradient(135deg, rgba(109, 94, 252, 0.09), rgba(34, 211, 238, 0.05));
    border: 1px solid rgba(109, 94, 252, 0.28);
    border-radius: 14px;
    padding: 18px 20px; margin-bottom: 6px;
  }
  .setting.target label { font-size: 15px; font-weight: 700; }

  /* Inputs */
  input[type="number"], select {
    background: rgba(13, 18, 32, 0.7);
    border: 1px solid rgba(120, 140, 200, 0.18);
    color: var(--text);
    padding: 10px 14px;
    border-radius: 10px;
    font: inherit; font-size: 14px;
    outline: none;
    transition: border-color .2s, box-shadow .2s;
  }
  input[type="number"]:focus, select:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(109, 94, 252, 0.15);
  }
  input[type="number"] { width: 120px; text-align: center; font-weight: 600; }
  input[type="number"]::placeholder { color: var(--text-dim); font-weight: 400; }
  .input-group { display: flex; gap: 8px; align-items: center; }
  .input-group .unit { color: var(--text-dim); font-size: 13px; }

  /* Chips */
  .chip-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; max-width: 420px; }
  .chip {
    background: rgba(13, 18, 32, 0.7);
    border: 1px solid rgba(120, 140, 200, 0.18);
    color: var(--text-dim);
    padding: 8px 14px; border-radius: 999px;
    font-size: 13px; font-weight: 500; cursor: pointer;
    transition: all .2s;
  }
  .chip:hover { border-color: var(--accent); color: var(--text); }
  .chip.active {
    background: linear-gradient(135deg, var(--accent), #8b7fff);
    border-color: transparent; color: white;
    box-shadow: 0 6px 20px -8px var(--accent);
  }

  /* Slider */
  .slider-wrap { position: relative; width: 320px; }
  input[type="range"] {
    -webkit-appearance: none; appearance: none;
    width: 100%; height: 8px; border-radius: 999px;
    background: rgba(120, 140, 200, 0.15);
    outline: none; cursor: pointer;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 22px; height: 22px; border-radius: 50%;
    background: linear-gradient(135deg, var(--accent-2), var(--accent));
    cursor: pointer;
    box-shadow: 0 2px 12px rgba(109, 94, 252, 0.6), 0 0 0 4px rgba(109, 94, 252, 0.12);
    transition: transform .15s;
  }
  input[type="range"]::-webkit-slider-thumb:hover { transform: scale(1.12); }
  input[type="range"]::-moz-range-thumb {
    width: 22px; height: 22px; border-radius: 50%; border: none;
    background: linear-gradient(135deg, var(--accent-2), var(--accent));
    cursor: pointer;
  }
  .value-badge {
    display: inline-block; font-family: 'JetBrains Mono', monospace;
    font-size: 12px; padding: 2px 8px; border-radius: 6px;
    background: rgba(109, 94, 252, 0.14);
    color: #b3aaff; margin-left: 8px; font-weight: 600;
  }

  /* Grid */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .grid-2 .setting { grid-template-columns: 1fr auto; border-bottom: none; }

  /* Toggle switch */
  .toggle { display: flex; align-items: center; gap: 14px; }
  .switch { position: relative; display: inline-block; width: 46px; height: 26px; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider-knob {
    position: absolute; inset: 0; cursor: pointer;
    background: rgba(120, 140, 200, 0.2);
    border-radius: 26px; transition: background .3s;
  }
  .slider-knob::before {
    content: ""; position: absolute;
    width: 20px; height: 20px; border-radius: 50%;
    background: white; left: 3px; top: 3px;
    transition: transform .3s;
  }
  .switch input:checked + .slider-knob {
    background: linear-gradient(135deg, var(--accent-2), var(--accent));
  }
  .switch input:checked + .slider-knob::before {
    transform: translateX(20px);
  }

  /* Primary button */
  .btn-primary {
    width: 100%;
    padding: 20px;
    border: none; border-radius: var(--radius);
    background: linear-gradient(135deg, var(--accent), #8b7fff 55%, var(--accent-2));
    color: white;
    font: inherit; font-size: 17px; font-weight: 700;
    letter-spacing: 0.2px; cursor: pointer;
    transition: all .25s;
    box-shadow: 0 20px 40px -18px var(--accent), 0 6px 16px -6px rgba(34, 211, 238, 0.4);
    position: relative; overflow: hidden;
  }
  .btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 26px 50px -18px var(--accent), 0 10px 22px -8px rgba(34, 211, 238, 0.5);
  }
  .btn-primary:disabled {
    opacity: 0.35; cursor: not-allowed;
    background: rgba(109, 94, 252, 0.3);
    box-shadow: none;
  }
  .btn-primary .spinner {
    display: inline-block; width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: white; border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 8px; vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Results */
  .results { display: none; }
  .results.active { display: block; animation: fadeUp .5s ease; }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .result-item {
    background: rgba(13, 18, 32, 0.6);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 10px;
  }
  .result-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px;
  }
  .result-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px; color: var(--text);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .result-badge {
    font-size: 11px; font-weight: 700;
    padding: 4px 9px; border-radius: 999px;
    background: rgba(34, 197, 94, 0.12);
    color: #86efac;
    border: 1px solid rgba(34, 197, 94, 0.25);
    flex-shrink: 0; margin-left: 10px;
  }
  .result-stats {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 12px; color: var(--text-dim);
    gap: 12px;
  }
  .dl-btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 8px 14px; border-radius: 9px;
    background: rgba(109, 94, 252, 0.14);
    color: #b3aaff;
    border: 1px solid rgba(109, 94, 252, 0.28);
    font-size: 12px; font-weight: 600; text-decoration: none;
    transition: all .2s;
  }
  .dl-btn:hover { background: rgba(109, 94, 252, 0.25); color: white; }
  .error-item {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    color: #fca5a5;
    padding: 12px 16px; border-radius: 10px;
    font-size: 13px; margin-bottom: 8px;
  }

  /* Footer */
  .foot {
    display: flex; justify-content: space-between;
    font-size: 12px; color: var(--text-dim);
    padding-top: 40px; opacity: 0.6;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb {
    background: rgba(120, 140, 200, 0.2);
    border-radius: 999px;
  }

  @media (max-width: 720px) {
    .nav { padding: 14px 18px; }
    .nav-links { display: none; }
    .container { padding: 40px 16px 60px; }
    .grid-2 { grid-template-columns: 1fr; }
    .setting { grid-template-columns: 1fr; text-align: left; }
    .chip-row { justify-content: flex-start; }
    .slider-wrap { width: 100%; }
  }
</style>
</head>
<body>
  <div class="orbs">
    <div class="orb a"></div>
    <div class="orb b"></div>
    <div class="orb c"></div>
  </div>

  <nav class="nav">
    <div class="brand">
      
      <span>Compressor<strong>.ai</strong></span>
    </div>
    <div class="nav-links">
      
    </div>
  </nav>

  <main class="container">
    <section class="hero">
     
      <h1>Compress to <span class="grad">any size</span><br>you want.</h1>
      <p class="sub">Drop images, type a target in KB or MB, and let the binary-search engine find the highest quality that still fits.</p>
    </section>

    <!-- Dropzone -->
    <div id="dropzone" class="dropzone">
      <input type="file" id="fileInput" multiple accept=".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff" hidden />
      <div class="dz-icon">⬆</div>
      <h3>Drop images here</h3>
      <p>or <button type="button" id="browseBtn" class="link-btn">browse files</button></p>
      <small>JPG · PNG · WEBP · BMP · TIFF — any size</small>
    </div>

    <!-- Files panel -->
    <div id="filesPanel" class="files-panel">
      <div class="files-header">
        <h3 id="filesCount">0 files</h3>
        <button id="clearBtn" class="clear-btn">Clear all</button>
      </div>
      <div id="fileList" class="file-list"></div>
    </div>

    <!-- Settings -->
    <section class="panel">
      <div class="panel-header">
        <h2>Compression Settings</h2>
      </div>

      <!-- ⭐ TARGET SIZE -->
      <div class="setting target">
        <div class="setting-info">
          <label> Target file size</label>
          <p class="hint">Leave blank to compress using preset/quality only</p>
        </div>
        <div class="input-group">
          <input type="number" id="targetSize" placeholder="e.g. 200" min="1" step="1" />
          <select id="targetUnit">
            <option value="KB" selected>KB</option>
            <option value="MB">MB</option>
          </select>
        </div>
      </div>

      <!-- Preset -->
      <div class="setting">
        <div class="setting-info">
          <label>Quality preset</label>
          <p class="hint">Click to auto-configure settings</p>
        </div>
        <div id="presetChips" class="chip-row">
          <button class="chip" data-preset="lossless">Lossless</button>
          <button class="chip" data-preset="high">High</button>
          <button class="chip active" data-preset="balanced">Balanced</button>
          <button class="chip" data-preset="web">Web</button>
          <button class="chip" data-preset="thumbnail">Thumbnail</button>
          <button class="chip" data-preset="aggressive">Aggressive</button>
        </div>
      </div>

      <!-- Quality -->
      <div class="setting">
        <div class="setting-info">
          <label>Quality <span id="qualityVal" class="value-badge">78</span></label>
          <p class="hint">Lower = smaller file · higher = sharper</p>
        </div>
        <div class="slider-wrap">
          <input type="range" id="quality" min="1" max="100" value="78" />
        </div>
      </div>

      <!-- Grid -->
      <div class="grid-2">
        <div class="setting compact">
          <div class="setting-info">
            <label>Max dimension</label>
            <p class="hint">Blank = keep original</p>
          </div>
          <div class="input-group">
            <input type="number" id="maxDim" placeholder="1920" min="16" />
            <span class="unit">px</span>
          </div>
        </div>
        <div class="setting compact">
          <div class="setting-info">
            <label>Output format</label>
            <p class="hint">WebP saves ~30% more</p>
          </div>
          <select id="format">
            <option value="">Keep original</option>
            <option value="JPEG">JPEG</option>
            <option value="PNG">PNG</option>
            <option value="WEBP">WEBP</option>
            <option value="BMP">BMP</option>
            <option value="TIFF">TIFF</option>
          </select>
        </div>
      </div>

      <!-- EXIF -->
      <div class="setting">
        <div class="setting-info">
          <label>Preserve EXIF metadata</label>
          <p class="hint">Camera info, GPS, date</p>
        </div>
        <label class="switch">
          <input type="checkbox" id="keepExif" checked />
          <span class="slider-knob"></span>
        </label>
      </div>
    </section>

    <!-- Compress -->
    <button id="compressBtn" class="btn-primary" disabled>Compress Now</button>

    <!-- Results -->
    <section id="results" class="results">
      <div class="panel" style="margin-top:26px">
        <div class="panel-header">
          <h2>✅ Results</h2>
          <div id="totalSaved" class="pill-sm success">—</div>
        </div>
        <div id="resultList"></div>
      </div>
    </section>

    <footer class="foot">
      <span>FastAPI · Pillow · Vanilla JS</span>
      <span>© 2026</span>
    </footer>
  </main>

<script>
(() => {
  const $ = (s) => document.querySelector(s);
  const files = [];
  const PRESETS = {
    lossless:   { quality: 100, maxDimension: null },
    high:       { quality: 90,  maxDimension: null },
    balanced:   { quality: 78,  maxDimension: null },
    web:        { quality: 65,  maxDimension: null },
    thumbnail:  { quality: 50,  maxDimension: 600 },
    aggressive: { quality: 35,  maxDimension: null },
  };

  const dropzone = $('#dropzone');
  const fileInput = $('#fileInput');
  const browseBtn = $('#browseBtn');
  const filesPanel = $('#filesPanel');
  const fileListEl = $('#fileList');
  const filesCount = $('#filesCount');
  const clearBtn = $('#clearBtn');
  const qualityEl = $('#quality');
  const qualityVal = $('#qualityVal');
  const targetSizeEl = $('#targetSize');
  const targetUnitEl = $('#targetUnit');
  const maxDimEl = $('#maxDim');
  const formatEl = $('#format');
  const keepExifEl = $('#keepExif');
  const compressBtn = $('#compressBtn');
  const resultsEl = $('#results');
  const resultListEl = $('#resultList');
  const totalSaved = $('#totalSaved');
  const presetChips = document.querySelectorAll('.chip');
  let activePreset = 'balanced';

  // ── File handling
  const fmtSize = (b) => {
    if (b < 1024) return b + ' B';
    if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB';
    return (b / (1024 * 1024)).toFixed(2) + ' MB';
  };

  const renderFiles = () => {
    filesCount.textContent = `${files.length} file${files.length === 1 ? '' : 's'}`;
    filesPanel.classList.toggle('active', files.length > 0);
    fileListEl.innerHTML = files.map((f, i) => `
      <div class="file-item">
        <span class="file-name">${f.name}</span>
        <span class="file-size">${fmtSize(f.size)}</span>
      </div>
    `).join('');
    compressBtn.disabled = files.length === 0;
  };

  const addFiles = (list) => {
    for (const f of list) {
      if (/\.(jpe?g|png|webp|bmp|tiff?)$/i.test(f.name)) files.push(f);
    }
    renderFiles();
  };

  dropzone.addEventListener('click', (e) => {
    if (e.target === browseBtn) return;
    fileInput.click();
  });
  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });
  fileInput.addEventListener('change', (e) => {
    addFiles(e.target.files);
    fileInput.value = '';
  });

  ['dragenter', 'dragover'].forEach(ev =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault(); dropzone.classList.add('drag');
    })
  );
  ['dragleave', 'drop'].forEach(ev =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault(); dropzone.classList.remove('drag');
    })
  );
  dropzone.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));

  clearBtn.addEventListener('click', () => {
    files.length = 0; renderFiles();
    resultsEl.classList.remove('active');
  });

  // ── Preset chips
  presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      presetChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      activePreset = chip.dataset.preset;
      const p = PRESETS[activePreset];
      if (p) {
        qualityEl.value = p.quality;
        qualityVal.textContent = p.quality;
        maxDimEl.value = p.maxDimension ?? '';
      }
    });
  });

  qualityEl.addEventListener('input', (e) => {
    qualityVal.textContent = e.target.value;
  });

  // ── Compress
  compressBtn.addEventListener('click', async () => {
    if (files.length === 0) return;

    const targetVal = parseFloat(targetSizeEl.value);
    const unit = targetUnitEl.value;
    const maxSizeKb = !isNaN(targetVal) && targetVal > 0
      ? (unit === 'MB' ? targetVal * 1024 : targetVal)
      : null;

    const maxDim = maxDimEl.value ? parseInt(maxDimEl.value) : null;
    const outputFormat = formatEl.value || null;
    const keepExif = keepExifEl.checked;
    const quality = parseInt(qualityEl.value);

    compressBtn.disabled = true;
    compressBtn.innerHTML = '<span class="spinner"></span>Compressing…';
    resultsEl.classList.remove('active');
    resultListEl.innerHTML = '';

    const allResults = [];
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      const form = new FormData();
      form.append('file', f);
      form.append('preset', activePreset);
      form.append('quality', quality);
      if (maxSizeKb) form.append('max_size_kb', maxSizeKb);
      if (maxDim) form.append('max_dimension', maxDim);
      if (outputFormat) form.append('output_format', outputFormat);
      form.append('keep_exif', keepExif);

      try {
        const res = await fetch('/api/compress', { method: 'POST', body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Server error');
        allResults.push({ ok: true, data, file: f });
      } catch (err) {
        allResults.push({ ok: false, error: err.message, file: f });
      }
      // render progressively
      renderResults(allResults);
    }

    const totalOrig = allResults.filter(r => r.ok).reduce((s, r) => s + r.data.stats.original_kb, 0);
    const totalComp = allResults.filter(r => r.ok).reduce((s, r) => s + r.data.stats.compressed_kb, 0);
    if (totalOrig > 0) {
      const saved = (1 - totalComp / totalOrig) * 100;
      totalSaved.textContent = `Saved ${saved.toFixed(1)}% overall`;
    } else {
      totalSaved.textContent = '—';
    }

    resultsEl.classList.add('active');
    compressBtn.disabled = false;
    compressBtn.textContent = '🚀 Compress Now';
  });

  function renderResults(results) {
    resultListEl.innerHTML = results.map(r => {
      if (!r.ok) {
        return `<div class="error-item">✗ ${r.file.name} — ${r.error}</div>`;
      }
      const s = r.data.stats;
      return `
        <div class="result-item">
          <div class="result-head">
            <span class="result-name">${r.data.filename}</span>
            <span class="result-badge">-${s.reduction_pct.toFixed(1)}%</span>
          </div>
          <div class="result-stats">
            <span>${s.original_kb.toFixed(1)} KB → ${s.compressed_kb.toFixed(1)} KB · ${s.format}</span>
            <a class="dl-btn" href="${r.data.download_url}" download>⬇ Download</a>
          </div>
        </div>
      `;
    }).join('');
  }

  // seed UI
  qualityVal.textContent = qualityEl.value;
})();
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════
#  LAUNCHER
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Image Compressor 2026")
    print("   → http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)