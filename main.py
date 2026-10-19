"""
Image Compressor 2026 — single-file web app
Run:  python app.py
Open: http://127.0.0.1:8000
"""

import os
import io
import uuid
import shutil
import subprocess
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pdf2docx import Converter

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
    # NOTE: Do NOT flatten RGBA/LA/P onto a white background here —
    # that was silently changing colours. We keep the image's own
    # palette / alpha and let the format-specific save step decide.
    if max_dimension:
        w, h = img.size
        longest = max(w, h)
        if longest > max_dimension:
            r = max_dimension / longest
            img = img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.LANCZOS)
    return img, exif_bytes


def _reduce_pixel_quality(img, quality):
    """
    Reduce 'pixel quality' by quantising the palette (colour depth),
    NOT by flattening transparency or shifting hues.
    """
    if quality >= 95:
        return img

    if quality <= 0:
        return img
    palette_size = max(8, min(256, int(8 + (quality / 94.0) * 248)))

    if img.mode in ("RGBA", "LA", "P"):
        try:
            if img.mode == "P":
                img = img.convert("RGBA")
            q = img.convert("RGBA").quantize(
                colors=palette_size, method=Image.FASTOCTREE
            )
            return q
        except Exception:
            return img

    if img.mode in ("RGB", "L"):
        try:
            return img.quantize(
                colors=palette_size,
                method=Image.MEDIANCUT,
                dither=Image.FLOYDSTEINBERG if quality >= 40 else Image.NONE,
            )
        except Exception:
            return img

    try:
        return img.convert("RGB").quantize(
            colors=palette_size, method=Image.MEDIANCUT
        )
    except Exception:
        return img


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
    save_img = img
    if fmt in ("JPEG", "BMP") and save_img.mode not in ("RGB", "L"):
        try:
            rgba = save_img.convert("RGBA")
            bg_color = (255, 255, 255)
            small = rgba.resize((32, 32))
            counts = {}
            for px in small.getdata():
                if px[3] >= 128:
                    counts[px[:3]] = counts.get(px[:3], 0) + 1
            if counts:
                bg_color = max(counts.items(), key=lambda kv: kv[1])[0]
            bg = Image.new("RGB", rgba.size, bg_color)
            bg.paste(rgba, mask=rgba.split()[-1])
            save_img = bg
        except Exception:
            save_img = save_img.convert("RGB")
    elif fmt not in ("PNG", "WEBP", "TIFF") and save_img.mode == "P":
        save_img = save_img.convert("RGB")

    buf = io.BytesIO()
    save_img.save(buf, format=fmt, **kw)
    return buf.getvalue(), save_img


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

    img = _reduce_pixel_quality(img, quality)

    save_kw = _save_kwargs(fmt, quality, subsampling, optimize, True, exif_bytes)

    if max_size_kb:
        target = max_size_kb * 1024
        best_bytes, best_q = None, 1
        best_img = img
        low, high = 5, 95
        for _ in range(12):
            q = (low + high) // 2
            trial = dict(save_kw)
            if fmt in ("JPEG", "WEBP"):
                trial["quality"] = q
            elif fmt == "PNG":
                trial["compress_level"] = max(0, min(9, int((100 - q) / 11)))
            trial_img = _reduce_pixel_quality(img, q)
            try:
                data, saved_img = _encode(trial_img, fmt, trial)
            except Exception:
                break
            if len(data) <= target:
                best_bytes, best_q, best_img = data, q, saved_img
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
                timg = _reduce_pixel_quality(timg, 40)
                trial = dict(save_kw)
                if fmt in ("JPEG", "WEBP"):
                    trial["quality"] = 40
                data, saved_img = _encode(timg, fmt, trial)
                if len(data) <= target:
                    best_bytes, best_img = data, saved_img
                    break
                scale -= 0.1
        if best_bytes is None:
            raise RuntimeError("Could not reach target size.")
        out_bytes = best_bytes
        img = best_img
    else:
        out_bytes, img = _encode(img, fmt, save_kw)

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
#  PDF → WORD CONVERTER
#  Now supports TWO engines:
#    • "pdf2docx"    → pure-python library (default, works everywhere)
#    • "libreoffice" → uses the system's `soffice` binary (better flow)
# ══════════════════════════════════════════════════════════════
def _find_soffice():
    """Locate the LibreOffice `soffice` binary across platforms."""
    # 1. Try PATH first
    for name in ("soffice", "soffice.exe", "libreoffice"):
        p = shutil.which(name)
        if p:
            return p
    # 2. Try common install locations
    candidates = [
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        "/usr/bin/soffice",
        "/usr/local/bin/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def pdf_to_word_bytes(input_bytes, filename, engine="pdf2docx"):
    """Convert a PDF byte-stream into a .docx byte-stream.

    engine = "pdf2docx"    → original behaviour (unchanged)
    engine = "libreoffice" → shell out to LibreOffice headless converter
    """
    if engine == "libreoffice":
        return _pdf_to_word_libreoffice(input_bytes, filename)
    # Default: existing pdf2docx path (unchanged)
    return _pdf_to_word_pdf2docx(input_bytes, filename)


def _pdf_to_word_pdf2docx(input_bytes, filename):
    """Original converter — untouched logic."""
    tmp_in = os.path.join(OUTPUT_DIR, f"_{uuid.uuid4().hex}.pdf")
    tmp_out = os.path.join(OUTPUT_DIR, f"_{uuid.uuid4().hex}.docx")

    try:
        with open(tmp_in, "wb") as f:
            f.write(input_bytes)

        cv = Converter(tmp_in)
        cv.convert(tmp_out)
        cv.close()

        with open(tmp_out, "rb") as f:
            docx_bytes = f.read()

        base = os.path.splitext(filename)[0]
        out_name = f"{base}.docx"

        return docx_bytes, out_name, {
            "original_kb": len(input_bytes) / 1024,
            "output_kb": len(docx_bytes) / 1024,
            "engine": "pdf2docx",
        }
    finally:
        for p in (tmp_in, tmp_out):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def _pdf_to_word_libreoffice(input_bytes, filename):
    """Convert using LibreOffice in headless mode via subprocess."""
    soffice = _find_soffice()
    if not soffice:
        raise RuntimeError(
            "LibreOffice (soffice) not found on this system. "
            "Install LibreOffice or pick the 'pdf2docx' engine instead."
        )

    # Work in an isolated temp dir so LibreOffice's profile/user dir
    # doesn't clash with a running LibreOffice GUI instance.
    work_dir = tempfile.mkdtemp(prefix="lo_conv_", dir=OUTPUT_DIR)
    try:
        base = os.path.splitext(os.path.basename(filename))[0] or "output"
        safe_base = "".join(c for c in base if c.isalnum() or c in "._- ") or "output"
        in_path = os.path.join(work_dir, f"{safe_base}.pdf")
        out_path = os.path.join(work_dir, f"{safe_base}.docx")

        with open(in_path, "wb") as f:
            f.write(input_bytes)

        # Use a private user profile so concurrent conversions don't lock up
        profile_dir = os.path.join(work_dir, "lo_profile")
        os.makedirs(profile_dir, exist_ok=True)
        profile_uri = "file:///" + profile_dir.replace("\\", "/").lstrip("/")

        cmd = [
            soffice,
            f"-env:UserInstallation={profile_uri}",
            "--headless",
            "--norestore",
            "--nologo",
            "--nodefault",
            "--infilter=writer_pdf_import",
            "--convert-to", "docx:MS Word 2007 XML",
            "--outdir", work_dir,
            in_path,
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )

        if not os.path.exists(out_path):
            err = (proc.stderr or b"").decode("utf-8", "ignore").strip()
            out = (proc.stdout or b"").decode("utf-8", "ignore").strip()
            raise RuntimeError(
                "LibreOffice conversion produced no output. "
                f"stdout: {out or '-'} | stderr: {err or '-'}"
            )

        with open(out_path, "rb") as f:
            docx_bytes = f.read()

        out_name = f"{base}.docx"
        return docx_bytes, out_name, {
            "original_kb": len(input_bytes) / 1024,
            "output_kb": len(docx_bytes) / 1024,
            "engine": "libreoffice",
        }
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


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
#  PDF → WORD ENDPOINT (now accepts `engine`)
# ══════════════════════════════════════════════════════════════
@app.post("/api/pdf-to-word")
async def api_pdf_to_word(
    file: UploadFile = File(...),
    engine: str = Form("pdf2docx"),
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext != ".pdf":
        raise HTTPException(400, f"Only .pdf files are supported (got {ext})")

    engine = (engine or "pdf2docx").strip().lower()
    if engine not in ("pdf2docx", "libreoffice"):
        raise HTTPException(400, f"Unknown engine: {engine}")

    try:
        data = await file.read()
        out_bytes, out_name, stats = pdf_to_word_bytes(
            data, file.filename, engine=engine
        )
    except Exception as e:
        raise HTTPException(500, f"PDF→Word conversion failed: {e}")

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
    background:
      radial-gradient(circle at 20% 20%, #1b1f46 0%, transparent 45%),
      radial-gradient(circle at 80% 30%, #0f2b46 0%, transparent 45%),
      radial-gradient(circle at 50% 80%, #2a1740 0%, transparent 45%),
      linear-gradient(135deg,
        #07090f 0%,
        #0b0f22 20%,
        #10142e 40%,
        #0d1220 60%,
        #0a0e1c 80%,
        #07090f 100%);
    background-size: 100% 100%, 100% 100%, 100% 100%, 400% 400%;
    background-attachment: fixed;
    animation: bgShift 26s ease-in-out infinite;
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
    -webkit-font-smoothing: antialiased;
  }
  @keyframes bgShift {
    0%   { background-position: 0% 0%,   100% 0%,  50% 100%, 0% 50%; }
    50%  { background-position: 60% 40%, 40% 60%, 60% 30%,  100% 50%; }
    100% { background-position: 0% 0%,   100% 0%,  50% 100%, 0% 50%; }
  }

  .orbs {
    position: fixed; inset: 0; z-index: 0;
    overflow: hidden; pointer-events: none;
    animation: hueCycle 32s linear infinite;
  }
  .orb {
    position: absolute; border-radius: 50%;
    filter: blur(120px); opacity: 0.34;
    will-change: transform;
  }
  .orb.a {
    width: 620px; height: 620px; background: var(--accent);
    top: -220px; left: -180px;
    animation: driftA 24s ease-in-out infinite alternate;
  }
  .orb.b {
    width: 520px; height: 520px; background: var(--accent-2);
    bottom: -220px; right: -160px;
    animation: driftB 28s ease-in-out infinite alternate;
  }
  .orb.c {
    width: 480px; height: 480px; background: var(--accent-3);
    top: 40%; left: 40%; opacity: 0.24;
    animation: driftC 22s ease-in-out infinite alternate;
  }
  .orb.d {
    width: 400px; height: 400px; background: #8b7fff;
    top: 15%; right: 10%; opacity: 0.22;
    animation: driftD 30s ease-in-out infinite alternate;
  }

  @keyframes driftA {
    0%   { transform: translate(0, 0)         scale(1)   rotate(0deg); }
    50%  { transform: translate(120px, 90px)  scale(1.15) rotate(15deg); }
    100% { transform: translate(-40px, 140px) scale(1.05) rotate(-8deg); }
  }
  @keyframes driftB {
    0%   { transform: translate(0, 0)          scale(1)    rotate(0deg); }
    50%  { transform: translate(-140px, -80px) scale(1.12) rotate(-20deg); }
    100% { transform: translate(60px, -140px)  scale(1.08) rotate(10deg); }
  }
  @keyframes driftC {
    0%   { transform: translate(0, 0)         scale(1)   rotate(0deg); }
    50%  { transform: translate(-90px, 60px)  scale(1.2) rotate(30deg); }
    100% { transform: translate(100px, -50px) scale(1.1) rotate(-15deg); }
  }
  @keyframes driftD {
    0%   { transform: translate(0, 0)           scale(1)   rotate(0deg); }
    50%  { transform: translate(80px, 120px)    scale(1.18) rotate(-25deg); }
    100% { transform: translate(-100px, 40px)   scale(1.06) rotate(18deg); }
  }
  @keyframes hueCycle {
    0%   { filter: hue-rotate(0deg); }
    50%  { filter: hue-rotate(35deg); }
    100% { filter: hue-rotate(0deg); }
  }

  @media (prefers-reduced-motion: reduce) {
    body, .orbs, .orb.a, .orb.b, .orb.c, .orb.d {
      animation: none !important;
    }
  }

  .nav {
    position: sticky; top: 0; z-index: 10;
    display: flex; justify-content: space-between; align-items: center;
    padding: 16px 32px;
    backdrop-filter: blur(22px) saturate(140%);
    background: rgba(7, 9, 15, 0.35);
    border-bottom: 1px solid var(--panel-border);
  }
  .brand { display: flex; align-items: center; gap: 12px; font-weight: 700; font-size: 17px; letter-spacing: -0.3px; }
  .logo {
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: grid; place-items: center;
    font-size: 18px; font-weight: 900;
  }
  .brand strong {
    background: linear-gradient(90deg, var(--accent-2), var(--accent));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .nav-links { display: flex; align-items: center; gap: 22px; font-size: 14px; color: var(--text-dim); }
  .nav-links a { color: var(--text-dim); text-decoration: none; transition: color .2s; }
  .nav-links a:hover { color: var(--text); }

  .container {
    position: relative; z-index: 1;
    max-width: 980px; margin: 0 auto;
    padding: 60px 24px 80px;
  }

  .hero { text-align: center; margin-bottom: 48px; }
  h1 {
    font-size: clamp(32px, 5vw, 54px); line-height: 1.05;
    font-weight: 800; letter-spacing: -1.5px; margin-bottom: 18px;
  }
  .grad {
    background: linear-gradient(90deg, #22d3ee, #6d5efc 45%, #f472b6);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .sub { font-size: 17px; color: var(--text-dim); max-width: 640px; margin: 0 auto; line-height: 1.55; }

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
    background: linear-gradient(180deg, #ef5350 0%, #d32f2f 100%);
    color: #fff;
    border: 1px solid #b71c1c;
    border-top: 1px solid #ff8a80;
    padding: 7px 14px; border-radius: 9px;
    font-size: 12px; font-weight: 700; cursor: pointer;
    letter-spacing: .3px;
    text-shadow: 0 -1px 0 rgba(0,0,0,.25);
    box-shadow:
      0 1px 0 rgba(255,255,255,.25) inset,
      0 -2px 0 rgba(0,0,0,.15) inset,
      0 3px 0 #7f1010,
      0 6px 12px rgba(0,0,0,.45);
    transition: transform .12s ease, box-shadow .12s ease, background .2s;
  }
  .clear-btn:hover {
    background: linear-gradient(180deg, #f0605a 0%, #e53e3e 100%);
  }
  .clear-btn:active {
    transform: translateY(3px);
    box-shadow:
      0 1px 0 rgba(255,255,255,.2) inset,
      0 -1px 0 rgba(0,0,0,.1) inset,
      0 0 0 #7f1010,
      0 2px 6px rgba(0,0,0,.4);
  }
  .file-list { display: flex; flex-direction: column; gap: 6px; max-height: 220px; overflow-y: auto; }
  .file-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 14px; border-radius: 10px;
    background: rgba(13, 18, 32, 0.6);
    font-size: 13px;
  }
  .file-name { color: var(--text); font-family: 'JetBrains Mono', monospace; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
  .file-size { color: var(--text-dim); font-size: 12px; margin-left: 12px; font-variant-numeric: tabular-nums; }

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

  .setting.target {
    background: linear-gradient(135deg, rgba(109, 94, 252, 0.09), rgba(34, 211, 238, 0.05));
    border: 1px solid rgba(109, 94, 252, 0.28);
    border-radius: 14px;
    padding: 18px 20px; margin-bottom: 6px;
  }
  .setting.target label { font-size: 15px; font-weight: 700; }

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

  .chip-row { display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; max-width: 420px; }
  .chip {
    position: relative;
    background: linear-gradient(180deg, #2a3355 0%, #1a2140 100%);
    color: #c7d0ec;
    padding: 9px 16px; border-radius: 999px;
    border: 1px solid rgba(120, 140, 200, 0.28);
    font-size: 13px; font-weight: 600; cursor: pointer;
    letter-spacing: .2px;
    text-shadow: 0 1px 0 rgba(0,0,0,.35);
    box-shadow:
      0 1px 0 rgba(255,255,255,.10) inset,
      0 -2px 0 rgba(0,0,0,.25) inset,
      0 3px 0 #0d1220,
      0 6px 14px rgba(0,0,0,.45);
    transition: transform .12s ease, box-shadow .12s ease, background .2s, color .2s;
  }
  .chip:hover {
    color: #fff;
    background: linear-gradient(180deg, #354070 0%, #202a55 100%);
    border-color: rgba(140, 160, 230, 0.45);
  }
  .chip:active {
    transform: translateY(3px);
    box-shadow:
      0 1px 0 rgba(255,255,255,.08) inset,
      0 -1px 0 rgba(0,0,0,.2) inset,
      0 0 0 #0d1220,
      0 2px 6px rgba(0,0,0,.4);
  }
  .chip.active {
    background: linear-gradient(180deg, #8b7fff 0%, #6d5efc 55%, #5646e0 100%);
    color: #fff;
    border-color: #a99bff;
    text-shadow: 0 1px 0 rgba(0,0,0,.3);
    box-shadow:
      0 1px 0 rgba(255,255,255,.35) inset,
      0 -2px 0 rgba(0,0,0,.22) inset,
      0 3px 0 #3a2fbf,
      0 4px 10px rgba(0,0,0,.4);
  }
  .chip.active:hover {
    background: linear-gradient(180deg, #9a8fff 0%, #7a6bff 55%, #6050ec 100%);
  }
  .chip.active:active {
    transform: translateY(3px);
    box-shadow:
      0 1px 0 rgba(255,255,255,.25) inset,
      0 -1px 0 rgba(0,0,0,.15) inset,
      0 0 0 #3a2fbf,
      0 2px 6px rgba(0,0,0,.4);
  }

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

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .grid-2 .setting { grid-template-columns: 1fr auto; border-bottom: none; }

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

  .btn-primary {
    position: relative;
    width: 100%;
    padding: 20px 24px;
    border: 1px solid #4a3fd1;
    border-top: 1px solid #b3aaff;
    border-bottom: 1px solid #3a2fbf;
    border-radius: var(--radius);
    background:
      linear-gradient(180deg,
        #9a8fff 0%,
        #6d5efc 35%,
        #5646e0 70%,
        #4737c5 100%);
    color: #fff;
    font: inherit; font-size: 17px; font-weight: 800;
    letter-spacing: 0.4px; cursor: pointer;
    text-shadow: 0 1px 2px rgba(0,0,0,.45), 0 -1px 0 rgba(0,0,0,.25);
    box-shadow:
      0 1px 0 rgba(255,255,255,.4) inset,
      0 2px 0 rgba(255,255,255,.12) inset,
      0 -3px 0 rgba(0,0,0,.18) inset,
      0 -1px 0 rgba(0,0,0,.35) inset,
      0 6px 0 #3a2fbf,
      0 7px 0 #2e2499,
      0 8px 18px rgba(0,0,0,.55);
    transform: translateY(0);
    transition:
      transform .12s cubic-bezier(.2,.7,.3,1),
      box-shadow .12s cubic-bezier(.2,.7,.3,1),
      background .2s;
    overflow: hidden;
    isolation: isolate;
  }
  .btn-primary::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 50%;
    background: linear-gradient(180deg, rgba(255,255,255,.22), rgba(255,255,255,0));
    border-radius: var(--radius) var(--radius) 40% 40% / var(--radius) var(--radius) 100% 100%;
    pointer-events: none;
    z-index: -1;
  }
  .btn-primary:hover:not(:disabled) {
    background:
      linear-gradient(180deg,
        #aa9fff 0%,
        #7a6bff 35%,
        #6050ec 70%,
        #5142d0 100%);
    transform: translateY(-1px);
    box-shadow:
      0 1px 0 rgba(255,255,255,.45) inset,
      0 2px 0 rgba(255,255,255,.14) inset,
      0 -3px 0 rgba(0,0,0,.18) inset,
      0 -1px 0 rgba(0,0,0,.35) inset,
      0 7px 0 #3a2fbf,
      0 8px 0 #2e2499,
      0 10px 22px rgba(0,0,0,.55);
  }
  .btn-primary:active:not(:disabled) {
    transform: translateY(6px);
    box-shadow:
      0 1px 0 rgba(255,255,255,.25) inset,
      0 -1px 0 rgba(0,0,0,.25) inset,
      0 0 0 #3a2fbf,
      0 1px 0 #2e2499,
      0 3px 8px rgba(0,0,0,.5);
    background:
      linear-gradient(180deg,
        #7f6ff0 0%,
        #5c4cf0 35%,
        #4a3ad0 70%,
        #3e30b8 100%);
  }
  .btn-primary:disabled {
    opacity: 0.55; cursor: not-allowed;
    background:
      linear-gradient(180deg, #4b4f6e 0%, #3a3e58 55%, #2e3149 100%);
    border-color: #2a2d40;
    border-top-color: #565a80;
    border-bottom-color: #1c1e2c;
    color: #aeb6d6;
    text-shadow: 0 1px 0 rgba(0,0,0,.5);
    box-shadow:
      0 1px 0 rgba(255,255,255,.1) inset,
      0 -2px 0 rgba(0,0,0,.25) inset,
      0 3px 0 #1c1e2c,
      0 3px 0 #14151f,
      0 8px 16px rgba(0,0,0,.45);
    transform: translateY(0);
  }
  .btn-primary .spinner {
    display: inline-block; width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: white; border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 8px; vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

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
    padding: 9px 16px; border-radius: 10px;
    background: linear-gradient(180deg, #3a3178 0%, #2b2460 55%, #221c4f 100%);
    color: #d4cdff;
    border: 1px solid #5449b8;
    border-top: 1px solid #7f73e0;
    border-bottom: 1px solid #1a1540;
    font-size: 12px; font-weight: 700; text-decoration: none;
    letter-spacing: .3px;
    text-shadow: 0 1px 0 rgba(0,0,0,.5);
    box-shadow:
      0 1px 0 rgba(255,255,255,.15) inset,
      0 -2px 0 rgba(0,0,0,.25) inset,
      0 3px 0 #14102e,
      0 6px 12px rgba(0,0,0,.5);
    transition: transform .12s ease, box-shadow .12s ease, background .2s, color .2s;
  }
  .dl-btn:hover {
    color: #fff;
    background: linear-gradient(180deg, #4a3f95 0%, #362d78 55%, #2a2366 100%);
  }
  .dl-btn:active {
    transform: translateY(3px);
    box-shadow:
      0 1px 0 rgba(255,255,255,.1) inset,
      0 -1px 0 rgba(0,0,0,.2) inset,
      0 0 0 #14102e,
      0 2px 6px rgba(0,0,0,.45);
  }
  .error-item {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    color: #fca5a5;
    padding: 12px 16px; border-radius: 10px;
    font-size: 13px; margin-bottom: 8px;
  }

  .tabs {
    display: flex; gap: 8px; margin-bottom: 26px;
    background:
      linear-gradient(180deg, #0a0d18 0%, #0f1426 100%);
    padding: 8px; border-radius: 16px;
    border: 1px solid var(--panel-border);
    box-shadow:
      0 1px 0 rgba(255,255,255,.04) inset,
      0 -2px 0 rgba(0,0,0,.4) inset,
      0 8px 20px rgba(0,0,0,.4);
  }
  .tab {
    position: relative;
    flex: 1; padding: 14px 18px; border-radius: 11px;
    background: linear-gradient(180deg, #1e2542 0%, #161b34 100%);
    color: #a4b0d2;
    border: 1px solid rgba(120, 140, 200, 0.18);
    font: inherit; font-weight: 700;
    font-size: 14px; cursor: pointer;
    letter-spacing: .3px;
    text-shadow: 0 1px 0 rgba(0,0,0,.4);
    box-shadow:
      0 1px 0 rgba(255,255,255,.06) inset,
      0 -2px 0 rgba(0,0,0,.3) inset,
      0 3px 0 #080a15,
      0 5px 10px rgba(0,0,0,.4);
    transition: transform .12s ease, box-shadow .12s ease, background .2s, color .2s;
  }
  .tab:hover { color: #fff; }
  .tab:active {
    transform: translateY(3px);
    box-shadow:
      0 1px 0 rgba(255,255,255,.05) inset,
      0 -1px 0 rgba(0,0,0,.2) inset,
      0 0 0 #080a15,
      0 2px 6px rgba(0,0,0,.4);
  }
  .tab.active {
    background:
      linear-gradient(180deg,
        #9a8fff 0%,
        #6d5efc 45%,
        #5646e0 75%,
        #4737c5 100%);
    color: #fff;
    border-color: #a99bff;
    border-top-color: #c9c0ff;
    border-bottom-color: #3a2fbf;
    text-shadow: 0 1px 0 rgba(0,0,0,.35);
    box-shadow:
      0 1px 0 rgba(255,255,255,.35) inset,
      0 -2px 0 rgba(0,0,0,.2) inset,
      0 3px 0 #3a2fbf,
      0 4px 10px rgba(0,0,0,.45);
  }
  .tab.active:hover {
    background:
      linear-gradient(180deg,
        #aa9fff 0%,
        #7a6bff 45%,
        #6050ec 75%,
        #5142d0 100%);
  }
  .tab.active:active {
    transform: translateY(3px);
    box-shadow:
      0 1px 0 rgba(255,255,255,.25) inset,
      0 -1px 0 rgba(0,0,0,.15) inset,
      0 0 0 #3a2fbf,
      0 2px 6px rgba(0,0,0,.4);
  }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; animation: fadeUp .35s ease; }

  .foot {
    display: flex; justify-content: space-between;
    font-size: 12px; color: var(--text-dim);
    padding-top: 40px; opacity: 0.6;
  }

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
    <div class="orb d"></div>
  </div>

  <nav class="nav">
    <div class="brand">
      <span>Image<strong>.Compressor</strong></span>
    </div>
    <div class="nav-links"></div>
  </nav>

  <main class="container">
    <section class="hero">
      <h1>Compress to <span class="grad">any size</span><br>you want.</h1>
      <p class="sub">Drop images, type a target in KB or MB, and let the binary-search engine find the highest quality that still fits.</p>
    </section>

    <div class="tabs">
      <button class="tab active" data-tab="compress">Image Compress</button>
      <button class="tab" data-tab="pdfword">PDF → Word</button>
    </div>

    <div class="tab-panel active" id="tab-compress">

      <div id="dropzone" class="dropzone">
        <input type="file" id="fileInput" multiple accept=".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff" hidden />
        <div class="dz-icon">⬆</div>
        <h3>Drop images here</h3>
        <p>or <button type="button" id="browseBtn" class="link-btn">browse files</button></p>
        <small>JPG · PNG · WEBP · BMP · TIFF — any size</small>
      </div>

      <div id="filesPanel" class="files-panel">
        <div class="files-header">
          <h3 id="filesCount">0 files</h3>
          <button id="clearBtn" class="clear-btn">Clear all</button>
        </div>
        <div id="fileList" class="file-list"></div>
      </div>

      <section class="panel">
        <div class="panel-header">
          <h2>Compression Settings</h2>
        </div>

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

        <div class="setting">
          <div class="setting-info">
            <label>Quality <span id="qualityVal" class="value-badge">78</span></label>
            <p class="hint">Lower = smaller file · higher = sharper</p>
          </div>
          <div class="slider-wrap">
            <input type="range" id="quality" min="1" max="100" value="78" />
          </div>
        </div>

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

      <button id="compressBtn" class="btn-primary" disabled>Compress Now</button>

      <section id="results" class="results">
        <div class="panel" style="margin-top:26px">
          <div class="panel-header">
            <h2>✅ Results</h2>
            <div id="totalSaved" class="pill-sm success">—</div>
          </div>
          <div id="resultList"></div>
        </div>
      </section>
    </div>

    <div class="tab-panel" id="tab-pdfword">

      <div id="pdfDropzone" class="dropzone">
        <input type="file" id="pdfFileInput" accept=".pdf,application/pdf" hidden />
        <div class="dz-icon">⬆</div>
        <h3>Drop a PDF file here</h3>
        <p>or <button type="button" id="pdfBrowseBtn" class="link-btn">browse files</button></p>
        <small>PDF only · best for text-based PDFs · scanned PDFs may need OCR first</small>
      </div>

      <div id="pdfFilePanel" class="files-panel">
        <div class="files-header">
          <h3 id="pdfFileCount">0 files</h3>
          <button id="pdfClearBtn" class="clear-btn">Clear</button>
        </div>
        <div id="pdfFileList" class="file-list"></div>
      </div>

      <!-- ═══════ NEW: CONVERSION ENGINE SELECTOR ═══════ -->
      <section class="panel">
        <div class="panel-header">
          <h2>Conversion Engine</h2>
        </div>
        <div class="setting">
          <div class="setting-info">
            <label>PDF → Word engine</label>
            <p class="hint">Choose which Engine converts the PDF</p>
          </div>
          <div id="engineChips" class="chip-row">
            <button class="chip active" data-engine="pdf2docx">Phoenix</button>
            <button class="chip" data-engine="libreoffice">Alkazar</button>
          </div>
        </div>
       
      </section>

      <button id="pdfConvertBtn" class="btn-primary" disabled>Convert PDF → Word</button>

      <section id="pdfResults" class="results">
        <div class="panel" style="margin-top:26px">
          <div class="panel-header">
            <h2>Results</h2>
          </div>
          <div id="pdfResultList"></div>
        </div>
      </section>
    </div>

    <footer class="foot">
      <span>© 2026 Sahil Pednekar</span>
    </footer>
  </main>

<script>
(() => {
  const $ = (s) => document.querySelector(s);

  // ═══════ TAB SWITCHING ═══════
  const tabs = document.querySelectorAll('.tab');
  const panels = {
    compress: $('#tab-compress'),
    pdfword:  $('#tab-pdfword'),
  };
  tabs.forEach(t => {
    t.addEventListener('click', () => {
      tabs.forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      Object.values(panels).forEach(p => p.classList.remove('active'));
      panels[t.dataset.tab].classList.add('active');
    });
  });

  // ═══════════════════════════════════════════════════════════
  //  TAB 1: IMAGE COMPRESSOR (unchanged)
  // ═══════════════════════════════════════════════════════════
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
  const presetChips = document.querySelectorAll('#presetChips .chip');
  let activePreset = 'balanced';

  const fmtSize = (b) => {
    if (b < 1024) return b + ' B';
    if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB';
    return (b / (1024 * 1024)).toFixed(2) + ' MB';
  };

  const renderFiles = () => {
    filesCount.textContent = `${files.length} file${files.length === 1 ? '' : 's'}`;
    filesPanel.classList.toggle('active', files.length > 0);
    fileListEl.innerHTML = files.map((f) => `
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
    compressBtn.textContent = 'Compress Now';
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

  qualityVal.textContent = qualityEl.value;

  // ═══════════════════════════════════════════════════════════
  //  TAB 2: PDF → WORD
  // ═══════════════════════════════════════════════════════════
  const pdfFiles = [];
  const pdfDropzone = $('#pdfDropzone');
  const pdfFileInput = $('#pdfFileInput');
  const pdfBrowseBtn = $('#pdfBrowseBtn');
  const pdfFilePanel = $('#pdfFilePanel');
  const pdfFileList = $('#pdfFileList');
  const pdfFileCount = $('#pdfFileCount');
  const pdfClearBtn = $('#pdfClearBtn');
  const pdfConvertBtn = $('#pdfConvertBtn');
  const pdfResultsEl = $('#pdfResults');
  const pdfResultListEl = $('#pdfResultList');

  // ═══════ NEW: ENGINE SELECTOR STATE ═══════
  const engineChips = document.querySelectorAll('#engineChips .chip');
  const engineHint = $('#engineHint');
  let activeEngine = 'pdf2docx';

  const ENGINE_HINTS = {
    pdf2docx:
      '<b>pdf2docx</b> · pure-Python, cross-platform, works out of the box. Best on ' +
      'simple single-column PDFs with tables.',
    libreoffice:
      '<b>LibreOffice</b> · uses the system <code>soffice</code> binary in headless mode. ' +
      'Usually better at flowing text and multi-column layouts. Requires LibreOffice installed ' +
      'on the machine running this server.',
  };

  engineChips.forEach(chip => {
    chip.addEventListener('click', () => {
      engineChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      activeEngine = chip.dataset.engine;
      engineHint.innerHTML = ENGINE_HINTS[activeEngine] || '';
    });
  });

  const renderPdfFiles = () => {
    pdfFileCount.textContent = `${pdfFiles.length} file${pdfFiles.length === 1 ? '' : 's'}`;
    pdfFilePanel.classList.toggle('active', pdfFiles.length > 0);
    pdfFileList.innerHTML = pdfFiles.map((f) => `
      <div class="file-item">
        <span class="file-name">${f.name}</span>
        <span class="file-size">${fmtSize(f.size)}</span>
      </div>
    `).join('');
    pdfConvertBtn.disabled = pdfFiles.length === 0;
  };

  const addPdfFiles = (list) => {
    for (const f of list) {
      if (/\.pdf$/i.test(f.name)) pdfFiles.push(f);
    }
    renderPdfFiles();
  };

  pdfDropzone.addEventListener('click', (e) => {
    if (e.target === pdfBrowseBtn) return;
    pdfFileInput.click();
  });
  pdfBrowseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    pdfFileInput.click();
  });
  pdfFileInput.addEventListener('change', (e) => {
    addPdfFiles(e.target.files);
    pdfFileInput.value = '';
  });

  ['dragenter', 'dragover'].forEach(ev =>
    pdfDropzone.addEventListener(ev, (e) => {
      e.preventDefault(); pdfDropzone.classList.add('drag');
    })
  );
  ['dragleave', 'drop'].forEach(ev =>
    pdfDropzone.addEventListener(ev, (e) => {
      e.preventDefault(); pdfDropzone.classList.remove('drag');
    })
  );
  pdfDropzone.addEventListener('drop', (e) => addPdfFiles(e.dataTransfer.files));

  pdfClearBtn.addEventListener('click', () => {
    pdfFiles.length = 0; renderPdfFiles();
    pdfResultsEl.classList.remove('active');
  });

  pdfConvertBtn.addEventListener('click', async () => {
    if (pdfFiles.length === 0) return;

    pdfConvertBtn.disabled = true;
    pdfConvertBtn.innerHTML =
      `<span class="spinner"></span>Converting via ${activeEngine === 'libreoffice' ? 'LibreOffice' : 'pdf2docx'}…`;
    pdfResultsEl.classList.remove('active');
    pdfResultListEl.innerHTML = '';

    const allResults = [];
    for (const f of pdfFiles) {
      const form = new FormData();
      form.append('file', f);
      form.append('engine', activeEngine);  // ═══════ NEW: send chosen engine ═══════
      try {
        const res = await fetch('/api/pdf-to-word', { method: 'POST', body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Server error');
        allResults.push({ ok: true, data, file: f });
      } catch (err) {
        allResults.push({ ok: false, error: err.message, file: f });
      }
      renderPdfResults(allResults);
    }

    pdfResultsEl.classList.add('active');
    pdfConvertBtn.disabled = false;
    pdfConvertBtn.textContent = '🔄 Convert PDF → Word';
  });

  function renderPdfResults(results) {
    pdfResultListEl.innerHTML = results.map(r => {
      if (!r.ok) {
        return `<div class="error-item">✗ ${r.file.name} — ${r.error}</div>`;
      }
      const s = r.data.stats;
      const engineLabel = (s.engine || activeEngine) === 'libreoffice' ? 'LO' : 'pdf2docx';
      return `
        <div class="result-item">
          <div class="result-head">
            <span class="result-name">${r.data.filename}</span>
            <span class="result-badge">DOCX · ${engineLabel}</span>
          </div>
          <div class="result-stats">
            <span>${s.original_kb.toFixed(1)} KB PDF → ${s.output_kb.toFixed(1)} KB DOCX</span>
            <a class="dl-btn" href="${r.data.download_url}" download>⬇ Download</a>
          </div>
        </div>
      `;
    }).join('');
  }

  engineHint.innerHTML = ENGINE_HINTS[activeEngine];
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
    print("\n Image Compressor Running Successfully")
    print("\n Click OR Copy Below Link To Start Using Our Service")
    print("\n You Can Also Copy The Below Link And Paste It On Normal Browser")
    print("   → http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)