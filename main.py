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
#  EMBEDDED HTML / CSS / JS — CLAY EFFECT UI + COLOUR THEME PICKER
# ══════════════════════════════════════════════════════════════
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Image Compressor · Clay Edition</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
  /* ═══════════════════════════════════════════════════════════
     THEME SYSTEM
     Each theme defines a full clay palette. The `data-theme`
     attribute on <html> switches every variable at once.
     ═══════════════════════════════════════════════════════════ */

  /* Default = Beige (original) */
  :root,
  [data-theme="beige"] {
    --clay-bg: #f0e9e2;
    --clay-bg-2: #e6dcd2;
    --clay-surface: #fcf8f3;
    --clay-surface-2: #f5ede4;
    --clay-surface-3: #ede2d6;
    --clay-text: #4a3f35;
    --clay-text-dim: #8a7460;
    --clay-text-soft: #9b8574;
    --clay-accent: #e8d5c0;
    --clay-accent-2: #d9d0c4;
    --clay-accent-3: #c7b7a5;
    --clay-shadow-dark: rgba(100, 70, 40, 0.4);
    --clay-shadow-light: rgba(255, 250, 240, 0.9);
    --clay-grad-1: #faf5ef;
    --clay-grad-2: #f5ebe0;
    --clay-grad-3: #c7a987;
    --clay-grad-4: #8a6f55;
    --clay-grad-5: #b89a7a;
    --clay-inset-a: rgba(220, 190, 160, 0.6);
    --clay-inset-b: rgba(220, 190, 160, 0.5);
    --clay-inset-c: rgba(220, 190, 160, 0.4);
    --clay-inset-d: rgba(200, 170, 140, 0.6);
    --clay-inset-e: rgba(255, 245, 235, 0.6);
    --clay-inset-f: rgba(255, 245, 235, 0.7);
    --clay-inset-g: rgba(255, 245, 235, 0.8);
    --clay-inset-h: rgba(200, 170, 140, 0.5);
    --clay-inset-i: rgba(220, 190, 160, 0.8);
    --clay-input-bg: #dac9b7;
    --clay-placeholder: #b8a898;
    --clay-success-bg: #e8ecd8;
    --clay-success-text: #6b8a5a;
    --clay-success-border: rgba(180, 200, 150, 0.6);
    --clay-error-bg: #f5e0d8;
    --clay-error-border: rgba(200, 130, 110, 0.4);
    --clay-error-text: #a05a4a;
    --clay-err-shadow: rgba(160, 90, 70, 0.3);
    --clay-err-inset-a: rgba(220, 190, 170, 0.5);
    --clay-err-inset-b: rgba(255, 245, 240, 0.8);
    --clay-link: #8a6f55;
    --clay-dz-border: rgba(160, 135, 110, 0.35);
    --clay-dz-hover: #b89a7a;
    --clay-clear-bg: #ede2d6;
    --clay-clear-text: #8a5a4a;
    --clay-clear-hover: #f0e0d0;
    --clay-clear-shadow: rgba(140, 90, 70, 0.35);
    --clay-clear-inset: rgba(200, 160, 140, 0.6);
    --clay-divider: rgba(160, 135, 110, 0.15);
    --clay-ring-1: #c7a987;
    --clay-ring-2: #b89a7a;
    --clay-ring-3: #9b7b5c;
    --clay-grad-text: linear-gradient(135deg, #c7a987, #8a6f55, #b89a7a);
    --clay-strong-bg: #f0e0d0;
    --clay-disabled-bg: #e0d6ca;
    --clay-disabled-text: #a89480;
    --clay-strong-text: #4a3f35;
    --clay-active-border: #d4bfa8;
    --clay-scroll-thumb: #c7b7a5;
    --clay-nav-bg: rgba(240, 233, 226, 0.7);
    --clay-nav-border: rgba(255, 245, 235, 0.6);
    --clay-body-grad-a: #f0e9e2;
    --clay-body-grad-b: #e6dcd2;
  }

  /* Mint / Sage */
  [data-theme="mint"] {
    --clay-bg: #e6f0e8;
    --clay-bg-2: #d6e6da;
    --clay-surface: #f4faf5;
    --clay-surface-2: #eaf3ec;
    --clay-surface-3: #d8e8de;
    --clay-text: #2f4739;
    --clay-text-dim: #5c7a68;
    --clay-text-soft: #7a9a86;
    --clay-accent: #b8dbc5;
    --clay-accent-2: #a5cdb4;
    --clay-accent-3: #8fbfa2;
    --clay-shadow-dark: rgba(60, 100, 75, 0.4);
    --clay-shadow-light: rgba(240, 255, 244, 0.9);
    --clay-grad-1: #f5fbf6;
    --clay-grad-2: #e0efe4;
    --clay-grad-3: #8fbfa2;
    --clay-grad-4: #5c7a68;
    --clay-grad-5: #a5cdb4;
    --clay-inset-a: rgba(180, 210, 190, 0.6);
    --clay-inset-b: rgba(180, 210, 190, 0.5);
    --clay-inset-c: rgba(180, 210, 190, 0.4);
    --clay-inset-d: rgba(160, 200, 175, 0.6);
    --clay-inset-e: rgba(240, 255, 244, 0.6);
    --clay-inset-f: rgba(240, 255, 244, 0.7);
    --clay-inset-g: rgba(240, 255, 244, 0.8);
    --clay-inset-h: rgba(160, 200, 175, 0.5);
    --clay-inset-i: rgba(180, 210, 190, 0.8);
    --clay-input-bg: #c8ded0;
    --clay-placeholder: #a5c0b0;
    --clay-success-bg: #dceadf;
    --clay-success-text: #4a8a5f;
    --clay-success-border: rgba(150, 200, 165, 0.6);
    --clay-error-bg: #f0e0d8;
    --clay-error-border: rgba(200, 130, 110, 0.4);
    --clay-error-text: #a05a4a;
    --clay-err-shadow: rgba(160, 90, 70, 0.3);
    --clay-err-inset-a: rgba(220, 190, 170, 0.5);
    --clay-err-inset-b: rgba(255, 245, 240, 0.8);
    --clay-link: #5c7a68;
    --clay-dz-border: rgba(120, 160, 135, 0.35);
    --clay-dz-hover: #8fbfa2;
    --clay-clear-bg: #d8e8de;
    --clay-clear-text: #7a4a4a;
    --clay-clear-hover: #e0f0e4;
    --clay-clear-shadow: rgba(120, 90, 70, 0.35);
    --clay-clear-inset: rgba(180, 200, 190, 0.6);
    --clay-divider: rgba(120, 160, 135, 0.15);
    --clay-ring-1: #8fbfa2;
    --clay-ring-2: #a5cdb4;
    --clay-ring-3: #6ba07e;
    --clay-grad-text: linear-gradient(135deg, #8fbfa2, #5c7a68, #a5cdb4);
    --clay-strong-bg: #d0e8d8;
    --clay-disabled-bg: #d8e5dc;
    --clay-disabled-text: #93ab9c;
    --clay-strong-text: #2f4739;
    --clay-active-border: #a5cdb4;
    --clay-scroll-thumb: #8fbfa2;
    --clay-nav-bg: rgba(230, 240, 232, 0.7);
    --clay-nav-border: rgba(240, 255, 244, 0.6);
    --clay-body-grad-a: #e6f0e8;
    --clay-body-grad-b: #d6e6da;
  }

  /* Lavender */
  [data-theme="lavender"] {
    --clay-bg: #ece6f2;
    --clay-bg-2: #ddd4e8;
    --clay-surface: #f8f5fc;
    --clay-surface-2: #f0ebf7;
    --clay-surface-3: #e2d9ef;
    --clay-text: #3f3357;
    --clay-text-dim: #71648a;
    --clay-text-soft: #8b7da6;
    --clay-accent: #d4c7e8;
    --clay-accent-2: #c4b3dc;
    --clay-accent-3: #b0a0cc;
    --clay-shadow-dark: rgba(80, 60, 120, 0.4);
    --clay-shadow-light: rgba(250, 246, 255, 0.9);
    --clay-grad-1: #f8f5fc;
    --clay-grad-2: #ede5f5;
    --clay-grad-3: #b0a0cc;
    --clay-grad-4: #71648a;
    --clay-grad-5: #c4b3dc;
    --clay-inset-a: rgba(210, 195, 230, 0.6);
    --clay-inset-b: rgba(210, 195, 230, 0.5);
    --clay-inset-c: rgba(210, 195, 230, 0.4);
    --clay-inset-d: rgba(190, 175, 215, 0.6);
    --clay-inset-e: rgba(250, 246, 255, 0.6);
    --clay-inset-f: rgba(250, 246, 255, 0.7);
    --clay-inset-g: rgba(250, 246, 255, 0.8);
    --clay-inset-h: rgba(190, 175, 215, 0.5);
    --clay-inset-i: rgba(210, 195, 230, 0.8);
    --clay-input-bg: #d8cfe6;
    --clay-placeholder: #b0a3c0;
    --clay-success-bg: #e0ecd8;
    --clay-success-text: #6b8a5a;
    --clay-success-border: rgba(180, 200, 150, 0.6);
    --clay-error-bg: #f0dce0;
    --clay-error-border: rgba(200, 120, 130, 0.4);
    --clay-error-text: #a05a6a;
    --clay-err-shadow: rgba(160, 90, 110, 0.3);
    --clay-err-inset-a: rgba(220, 190, 200, 0.5);
    --clay-err-inset-b: rgba(255, 245, 250, 0.8);
    --clay-link: #71648a;
    --clay-dz-border: rgba(150, 130, 180, 0.35);
    --clay-dz-hover: #b0a0cc;
    --clay-clear-bg: #e2d9ef;
    --clay-clear-text: #7a5a8a;
    --clay-clear-hover: #ede4f7;
    --clay-clear-shadow: rgba(120, 90, 150, 0.35);
    --clay-clear-inset: rgba(200, 180, 220, 0.6);
    --clay-divider: rgba(150, 130, 180, 0.15);
    --clay-ring-1: #b0a0cc;
    --clay-ring-2: #c4b3dc;
    --clay-ring-3: #8a7aa6;
    --clay-grad-text: linear-gradient(135deg, #b0a0cc, #71648a, #c4b3dc);
    --clay-strong-bg: #e4dbf0;
    --clay-disabled-bg: #ddd5e8;
    --clay-disabled-text: #9b8fb0;
    --clay-strong-text: #3f3357;
    --clay-active-border: #c4b3dc;
    --clay-scroll-thumb: #b0a0cc;
    --clay-nav-bg: rgba(236, 230, 242, 0.7);
    --clay-nav-border: rgba(250, 246, 255, 0.6);
    --clay-body-grad-a: #ece6f2;
    --clay-body-grad-b: #ddd4e8;
  }

  /* Peach / Coral */
  [data-theme="peach"] {
    --clay-bg: #f7e9e2;
    --clay-bg-2: #f0d9cd;
    --clay-surface: #fdf6f2;
    --clay-surface-2: #fbeee6;
    --clay-surface-3: #f4ddd0;
    --clay-text: #5a3d30;
    --clay-text-dim: #96705e;
    --clay-text-soft: #b08a76;
    --clay-accent: #f5cbb8;
    --clay-accent-2: #efb79f;
    --clay-accent-3: #e5a085;
    --clay-shadow-dark: rgba(140, 80, 55, 0.4);
    --clay-shadow-light: rgba(255, 248, 244, 0.9);
    --clay-grad-1: #fdf6f2;
    --clay-grad-2: #f8e8de;
    --clay-grad-3: #e5a085;
    --clay-grad-4: #96705e;
    --clay-grad-5: #efb79f;
    --clay-inset-a: rgba(240, 205, 185, 0.6);
    --clay-inset-b: rgba(240, 205, 185, 0.5);
    --clay-inset-c: rgba(240, 205, 185, 0.4);
    --clay-inset-d: rgba(230, 185, 165, 0.6);
    --clay-inset-e: rgba(255, 248, 244, 0.6);
    --clay-inset-f: rgba(255, 248, 244, 0.7);
    --clay-inset-g: rgba(255, 248, 244, 0.8);
    --clay-inset-h: rgba(230, 185, 165, 0.5);
    --clay-inset-i: rgba(240, 205, 185, 0.8);
    --clay-input-bg: #f0d0c0;
    --clay-placeholder: #c8a898;
    --clay-success-bg: #e8ecd8;
    --clay-success-text: #6b8a5a;
    --clay-success-border: rgba(180, 200, 150, 0.6);
    --clay-error-bg: #f5dcd8;
    --clay-error-border: rgba(200, 110, 100, 0.4);
    --clay-error-text: #a05a4a;
    --clay-err-shadow: rgba(160, 80, 60, 0.3);
    --clay-err-inset-a: rgba(220, 180, 170, 0.5);
    --clay-err-inset-b: rgba(255, 245, 240, 0.8);
    --clay-link: #96705e;
    --clay-dz-border: rgba(190, 140, 115, 0.35);
    --clay-dz-hover: #e5a085;
    --clay-clear-bg: #f4ddd0;
    --clay-clear-text: #8a4a3a;
    --clay-clear-hover: #fbe8dc;
    --clay-clear-shadow: rgba(160, 90, 70, 0.35);
    --clay-clear-inset: rgba(230, 185, 165, 0.6);
    --clay-divider: rgba(190, 140, 115, 0.15);
    --clay-ring-1: #e5a085;
    --clay-ring-2: #efb79f;
    --clay-ring-3: #c88064;
    --clay-grad-text: linear-gradient(135deg, #e5a085, #96705e, #efb79f);
    --clay-strong-bg: #fadccb;
    --clay-disabled-bg: #efdcd2;
    --clay-disabled-text: #b89a8a;
    --clay-strong-text: #5a3d30;
    --clay-active-border: #efb79f;
    --clay-scroll-thumb: #e5a085;
    --clay-nav-bg: rgba(247, 233, 226, 0.7);
    --clay-nav-border: rgba(255, 248, 244, 0.6);
    --clay-body-grad-a: #f7e9e2;
    --clay-body-grad-b: #f0d9cd;
  }

  /* Sky / Blue */
  [data-theme="sky"] {
    --clay-bg: #e6eef7;
    --clay-bg-2: #d3e0ee;
    --clay-surface: #f4f8fc;
    --clay-surface-2: #eaf1f8;
    --clay-surface-3: #d8e5f0;
    --clay-text: #2c3f57;
    --clay-text-dim: #5a7290;
    --clay-text-soft: #7890ac;
    --clay-accent: #b8cfe5;
    --clay-accent-2: #a5bedb;
    --clay-accent-3: #8fabc9;
    --clay-shadow-dark: rgba(55, 80, 115, 0.4);
    --clay-shadow-light: rgba(244, 250, 255, 0.9);
    --clay-grad-1: #f4f8fc;
    --clay-grad-2: #e2ecf5;
    --clay-grad-3: #8fabc9;
    --clay-grad-4: #5a7290;
    --clay-grad-5: #a5bedb;
    --clay-inset-a: rgba(185, 205, 225, 0.6);
    --clay-inset-b: rgba(185, 205, 225, 0.5);
    --clay-inset-c: rgba(185, 205, 225, 0.4);
    --clay-inset-d: rgba(165, 190, 215, 0.6);
    --clay-inset-e: rgba(244, 250, 255, 0.6);
    --clay-inset-f: rgba(244, 250, 255, 0.7);
    --clay-inset-g: rgba(244, 250, 255, 0.8);
    --clay-inset-h: rgba(165, 190, 215, 0.5);
    --clay-inset-i: rgba(185, 205, 225, 0.8);
    --clay-input-bg: #c8dae8;
    --clay-placeholder: #a0b4c8;
    --clay-success-bg: #e0ecd8;
    --clay-success-text: #6b8a5a;
    --clay-success-border: rgba(180, 200, 150, 0.6);
    --clay-error-bg: #f0dce0;
    --clay-error-border: rgba(200, 120, 130, 0.4);
    --clay-error-text: #a05a6a;
    --clay-err-shadow: rgba(160, 90, 110, 0.3);
    --clay-err-inset-a: rgba(220, 190, 200, 0.5);
    --clay-err-inset-b: rgba(255, 245, 250, 0.8);
    --clay-link: #5a7290;
    --clay-dz-border: rgba(115, 145, 180, 0.35);
    --clay-dz-hover: #8fabc9;
    --clay-clear-bg: #d8e5f0;
    --clay-clear-text: #4a5a7a;
    --clay-clear-hover: #e4eef8;
    --clay-clear-shadow: rgba(70, 95, 130, 0.35);
    --clay-clear-inset: rgba(180, 200, 220, 0.6);
    --clay-divider: rgba(115, 145, 180, 0.15);
    --clay-ring-1: #8fabc9;
    --clay-ring-2: #a5bedb;
    --clay-ring-3: #6b8aa6;
    --clay-grad-text: linear-gradient(135deg, #8fabc9, #5a7290, #a5bedb);
    --clay-strong-bg: #d6e6f2;
    --clay-disabled-bg: #d8e3ed;
    --clay-disabled-text: #93a8bc;
    --clay-strong-text: #2c3f57;
    --clay-active-border: #a5bedb;
    --clay-scroll-thumb: #8fabc9;
    --clay-nav-bg: rgba(230, 238, 247, 0.7);
    --clay-nav-border: rgba(244, 250, 255, 0.6);
    --clay-body-grad-a: #e6eef7;
    --clay-body-grad-b: #d3e0ee;
  }

  /* Rose / Pink */
  [data-theme="rose"] {
    --clay-bg: #f5e6ec;
    --clay-bg-2: #eed6df;
    --clay-surface: #fdf5f8;
    --clay-surface-2: #faeef3;
    --clay-surface-3: #f3dde6;
    --clay-text: #573243;
    --clay-text-dim: #90647a;
    --clay-text-soft: #ab7f94;
    --clay-accent: #f0c4d6;
    --clay-accent-2: #e8b0c6;
    --clay-accent-3: #db96b0;
    --clay-shadow-dark: rgba(120, 60, 90, 0.4);
    --clay-shadow-light: rgba(255, 248, 252, 0.9);
    --clay-grad-1: #fdf5f8;
    --clay-grad-2: #f8e5ee;
    --clay-grad-3: #db96b0;
    --clay-grad-4: #90647a;
    --clay-grad-5: #e8b0c6;
    --clay-inset-a: rgba(235, 200, 215, 0.6);
    --clay-inset-b: rgba(235, 200, 215, 0.5);
    --clay-inset-c: rgba(235, 200, 215, 0.4);
    --clay-inset-d: rgba(225, 180, 200, 0.6);
    --clay-inset-e: rgba(255, 248, 252, 0.6);
    --clay-inset-f: rgba(255, 248, 252, 0.7);
    --clay-inset-g: rgba(255, 248, 252, 0.8);
    --clay-inset-h: rgba(225, 180, 200, 0.5);
    --clay-inset-i: rgba(235, 200, 215, 0.8);
    --clay-input-bg: #ecd0dc;
    --clay-placeholder: #c4a0b4;
    --clay-success-bg: #e8ecd8;
    --clay-success-text: #6b8a5a;
    --clay-success-border: rgba(180, 200, 150, 0.6);
    --clay-error-bg: #f5dce0;
    --clay-error-border: rgba(200, 110, 120, 0.4);
    --clay-error-text: #a05a5a;
    --clay-err-shadow: rgba(160, 80, 80, 0.3);
    --clay-err-inset-a: rgba(220, 180, 185, 0.5);
    --clay-err-inset-b: rgba(255, 245, 245, 0.8);
    --clay-link: #90647a;
    --clay-dz-border: rgba(180, 130, 150, 0.35);
    --clay-dz-hover: #db96b0;
    --clay-clear-bg: #f3dde6;
    --clay-clear-text: #8a4a5a;
    --clay-clear-hover: #fbe8f0;
    --clay-clear-shadow: rgba(150, 80, 100, 0.35);
    --clay-clear-inset: rgba(230, 185, 200, 0.6);
    --clay-divider: rgba(180, 130, 150, 0.15);
    --clay-ring-1: #db96b0;
    --clay-ring-2: #e8b0c6;
    --clay-ring-3: #c07090;
    --clay-grad-text: linear-gradient(135deg, #db96b0, #90647a, #e8b0c6);
    --clay-strong-bg: #f8dce8;
    --clay-disabled-bg: #efdce4;
    --clay-disabled-text: #b89aa8;
    --clay-strong-text: #573243;
    --clay-active-border: #e8b0c6;
    --clay-scroll-thumb: #db96b0;
    --clay-nav-bg: rgba(245, 230, 236, 0.7);
    --clay-nav-border: rgba(255, 248, 252, 0.6);
    --clay-body-grad-a: #f5e6ec;
    --clay-body-grad-b: #eed6df;
  }

  /* Dark Clay / Charcoal */
  [data-theme="charcoal"] {
    --clay-bg: #1c1e24;
    --clay-bg-2: #14161b;
    --clay-surface: #2a2d36;
    --clay-surface-2: #23262e;
    --clay-surface-3: #1c1f26;
    --clay-text: #e8e4de;
    --clay-text-dim: #a8a49c;
    --clay-text-soft: #84807a;
    --clay-accent: #3a3e48;
    --clay-accent-2: #484c58;
    --clay-accent-3: #5a5e6c;
    --clay-shadow-dark: rgba(0, 0, 0, 0.6);
    --clay-shadow-light: rgba(80, 85, 96, 0.45);
    --clay-grad-1: #2f333d;
    --clay-grad-2: #1e2128;
    --clay-grad-3: #a8a49c;
    --clay-grad-4: #e8e4de;
    --clay-grad-5: #c4c0b8;
    --clay-inset-a: rgba(50, 55, 66, 0.8);
    --clay-inset-b: rgba(50, 55, 66, 0.6);
    --clay-inset-c: rgba(50, 55, 66, 0.5);
    --clay-inset-d: rgba(60, 65, 78, 0.7);
    --clay-inset-e: rgba(70, 76, 90, 0.5);
    --clay-inset-f: rgba(70, 76, 90, 0.6);
    --clay-inset-g: rgba(70, 76, 90, 0.7);
    --clay-inset-h: rgba(55, 60, 72, 0.6);
    --clay-inset-i: rgba(50, 55, 66, 0.7);
    --clay-input-bg: #14161b;
    --clay-placeholder: #6a6e78;
    --clay-success-bg: #2a3a2a;
    --clay-success-text: #86c690;
    --clay-success-border: rgba(120, 180, 130, 0.4);
    --clay-error-bg: #3a2428;
    --clay-error-border: rgba(200, 100, 110, 0.4);
    --clay-error-text: #f0a0a8;
    --clay-err-shadow: rgba(0, 0, 0, 0.5);
    --clay-err-inset-a: rgba(70, 40, 45, 0.6);
    --clay-err-inset-b: rgba(90, 55, 60, 0.5);
    --clay-link: #b0a8a0;
    --clay-dz-border: rgba(120, 120, 130, 0.35);
    --clay-dz-hover: #6a6e7a;
    --clay-clear-bg: #2e1f22;
    --clay-clear-text: #e09090;
    --clay-clear-hover: #3a282c;
    --clay-clear-shadow: rgba(0, 0, 0, 0.6);
    --clay-clear-inset: rgba(90, 50, 55, 0.5);
    --clay-divider: rgba(120, 125, 135, 0.15);
    --clay-ring-1: #a8a49c;
    --clay-ring-2: #c4c0b8;
    --clay-ring-3: #84807a;
    --clay-grad-text: linear-gradient(135deg, #c4c0b8, #e8e4de, #a8a49c);
    --clay-strong-bg: #3a3e48;
    --clay-disabled-bg: #24262d;
    --clay-disabled-text: #6a6e78;
    --clay-strong-text: #e8e4de;
    --clay-active-border: #5a5e6c;
    --clay-scroll-thumb: #5a5e6c;
    --clay-nav-bg: rgba(20, 22, 27, 0.75);
    --clay-nav-border: rgba(60, 65, 78, 0.5);
    --clay-body-grad-a: #1c1e24;
    --clay-body-grad-b: #14161b;
  }

  /* Shared constants */
  :root {
    --radius: 50px;
    --radius-sm: 30px;
    --radius-xs: 20px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    font-family: 'Quicksand', system-ui, sans-serif;
    background: var(--clay-bg);
    background-image:
      radial-gradient(circle at 20% 30%, var(--clay-grad-1) 0%, transparent 50%),
      radial-gradient(circle at 80% 70%, var(--clay-grad-2) 0%, transparent 50%),
      linear-gradient(135deg, var(--clay-body-grad-a) 0%, var(--clay-body-grad-b) 100%);
    background-attachment: fixed;
    color: var(--clay-text);
    min-height: 100vh;
    overflow-x: hidden;
    -webkit-font-smoothing: antialiased;
    padding: 0 0 60px;
    transition: background .35s ease, color .35s ease;
  }

  /* floating clay pebbles background */
  .pebble {
    position: fixed; border-radius: 50%;
    pointer-events: none; z-index: 0;
    opacity: 0.4;
    background: var(--clay-surface-2);
    box-shadow:
      0 20px 40px -15px var(--clay-shadow-dark),
      inset 0 -10px 15px var(--clay-inset-a),
      inset 0 10px 20px var(--clay-shadow-light);
    animation: floatPebble 12s ease-in-out infinite;
    transition: background .35s ease, box-shadow .35s ease;
  }
  .pebble.p1 { width: 200px; height: 200px; top: 5%; left: -60px; animation-delay: 0s; }
  .pebble.p2 { width: 140px; height: 140px; top: 60%; right: -40px; animation-delay: -4s; }
  .pebble.p3 { width: 100px; height: 100px; bottom: 10%; left: 10%; animation-delay: -8s; }
  .pebble.p4 { width: 260px; height: 260px; top: 25%; right: -120px; opacity: 0.25; animation-delay: -2s; }
  .pebble.p5 { width: 80px; height: 80px; top: 15%; left: 45%; opacity: 0.3; animation-delay: -6s; }

  @keyframes floatPebble {
    0%, 100% { transform: translateY(0) rotate(-3deg) scale(1); }
    50% { transform: translateY(-25px) rotate(3deg) scale(1.03); }
  }
  @media (prefers-reduced-motion: reduce) {
    .pebble { animation: none !important; }
  }

  .nav {
    position: sticky; top: 0; z-index: 10;
    display: flex; justify-content: space-between; align-items: center;
    padding: 18px 34px;
    backdrop-filter: blur(14px) saturate(140%);
    background: var(--clay-nav-bg);
    border-bottom: 1px solid var(--clay-nav-border);
    box-shadow: 0 4px 20px -10px var(--clay-shadow-dark);
    transition: background .35s ease, border-color .35s ease;
  }
  .brand {
    display: flex; align-items: center; gap: 14px;
    font-weight: 700; font-size: 18px; letter-spacing: -0.3px;
    color: var(--clay-text);
    text-shadow: 0 1px 2px var(--clay-shadow-light);
    transition: color .35s ease;
  }
  .logo {
    width: 42px; height: 42px; border-radius: 14px;
    background: var(--clay-surface-2);
    display: grid; place-items: center;
    font-size: 20px; font-weight: 900;
    box-shadow:
      0 10px 18px -8px var(--clay-shadow-dark),
      inset 0 -4px 6px var(--clay-inset-i),
      inset 0 4px 8px var(--clay-shadow-light);
    color: var(--clay-text);
    border: 1px solid var(--clay-inset-f);
    transition: background .35s ease, color .35s ease, box-shadow .35s ease;
  }
  .brand strong { color: var(--clay-text-soft); transition: color .35s ease; }

  /* ═══════════════════════════════════════════════════════════
     THEME SWATCH PICKER
     ═══════════════════════════════════════════════════════════ */
  .theme-picker {
    display: flex; align-items: center; gap: 10px;
  }
  .theme-label {
    font-size: 11px; font-weight: 700;
    color: var(--clay-text-soft);
    letter-spacing: 1.2px; text-transform: uppercase;
    margin-right: 2px;
    transition: color .35s ease;
  }
  .theme-swatches {
    display: flex; gap: 8px;
  }
  .theme-swatch {
    width: 30px; height: 30px; border-radius: 50%;
    cursor: pointer; border: 2px solid transparent;
    padding: 0; position: relative;
    transition: transform .18s cubic-bezier(.2,.7,.3,1), border-color .18s ease;
    box-shadow:
      0 6px 10px -4px var(--clay-shadow-dark),
      inset 0 -3px 5px rgba(0,0,0,0.18),
      inset 0 3px 6px rgba(255,255,255,0.35);
  }
  .theme-swatch:hover { transform: translateY(-2px) scale(1.08); }
  .theme-swatch.active {
    border-color: var(--clay-text);
    transform: scale(1.1);
    box-shadow:
      0 8px 14px -4px var(--clay-shadow-dark),
      0 0 0 3px var(--clay-surface-2),
      inset 0 -3px 5px rgba(0,0,0,0.18),
      inset 0 3px 6px rgba(255,255,255,0.35);
  }
  /* swatch fills */
  .sw-beige    { background: linear-gradient(135deg, #f0e9e2 0%, #c7b7a5 100%); }
  .sw-mint     { background: linear-gradient(135deg, #e6f0e8 0%, #8fbfa2 100%); }
  .sw-lavender { background: linear-gradient(135deg, #ece6f2 0%, #b0a0cc 100%); }
  .sw-peach    { background: linear-gradient(135deg, #f7e9e2 0%, #e5a085 100%); }
  .sw-sky      { background: linear-gradient(135deg, #e6eef7 0%, #8fabc9 100%); }
  .sw-rose     { background: linear-gradient(135deg, #f5e6ec 0%, #db96b0 100%); }
  .sw-charcoal { background: linear-gradient(135deg, #3a3e48 0%, #14161b 100%); }

  .container {
    position: relative; z-index: 1;
    max-width: 980px; margin: 0 auto;
    padding: 60px 24px 40px;
  }

  .hero { text-align: center; margin-bottom: 48px; }
  h1 {
    font-size: clamp(32px, 5vw, 54px); line-height: 1.05;
    font-weight: 700; letter-spacing: -1.5px; margin-bottom: 18px;
    color: var(--clay-text);
    text-shadow:
      0 2px 4px var(--clay-shadow-light),
      0 8px 14px var(--clay-shadow-dark);
    transition: color .35s ease, text-shadow .35s ease;
  }
  .grad {
    background: var(--clay-grad-text);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    text-shadow: none;
  }
  .sub {
    font-size: 17px; color: var(--clay-text-dim);
    max-width: 640px; margin: 0 auto; line-height: 1.55;
    text-shadow: 0 1px 1px var(--clay-shadow-light);
    transition: color .35s ease;
  }

  /* ───────── CLAY DROPZONE ───────── */
  .dropzone {
    border: 2px dashed var(--clay-dz-border);
    border-radius: var(--radius);
    padding: 56px 32px; text-align: center;
    background: var(--clay-surface-2);
    transition: all .25s ease, background .35s ease, border-color .35s ease;
    cursor: pointer; margin-bottom: 26px;
    position: relative; overflow: hidden;
    box-shadow:
      0 20px 30px -15px var(--clay-shadow-dark),
      inset 0 -8px 12px var(--clay-inset-b),
      inset 0 8px 16px var(--clay-shadow-light);
  }
  .dropzone:hover, .dropzone.drag {
    border-color: var(--clay-dz-hover);
    background: var(--clay-surface);
    transform: translateY(-2px);
    box-shadow:
      0 26px 36px -16px var(--clay-shadow-dark),
      inset 0 -8px 12px var(--clay-inset-a),
      inset 0 8px 18px var(--clay-shadow-light);
  }
  .dz-icon {
    font-size: 44px; margin-bottom: 12px;
    color: var(--clay-text-soft);
    filter: drop-shadow(2px 4px 6px var(--clay-shadow-dark));
    transition: color .35s ease;
  }
  .dropzone h3 { font-size: 21px; font-weight: 700; margin-bottom: 6px; color: var(--clay-text); transition: color .35s ease; }
  .dropzone p { color: var(--clay-text-dim); font-size: 14px; transition: color .35s ease; }
  .dropzone small { display: block; margin-top: 12px; color: var(--clay-text-soft); font-size: 12px; opacity: 0.8; transition: color .35s ease; }
  .link-btn {
    background: none; border: none; color: var(--clay-link);
    font: inherit; cursor: pointer; text-decoration: underline;
    text-underline-offset: 3px; font-weight: 700;
    transition: color .35s ease;
  }
  .link-btn:hover { color: var(--clay-text); }

  /* ───────── FILES PANEL ───────── */
  .files-panel {
    background: var(--clay-surface-2);
    border: 1px solid var(--clay-inset-f);
    border-radius: var(--radius-sm);
    padding: 20px 22px; margin-bottom: 22px;
    display: none;
    box-shadow:
      0 16px 24px -14px var(--clay-shadow-dark),
      inset 0 -6px 10px var(--clay-inset-b),
      inset 0 6px 12px var(--clay-shadow-light);
    transition: background .35s ease, border-color .35s ease;
  }
  .files-panel.active { display: block; }
  .files-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 14px;
  }
  .files-header h3 {
    font-size: 13px; font-weight: 700; color: var(--clay-text-soft);
    text-transform: uppercase; letter-spacing: 1.5px;
    transition: color .35s ease;
  }
  .clear-btn {
    background: var(--clay-clear-bg);
    color: var(--clay-clear-text);
    border: 1px solid var(--clay-inset-g);
    padding: 9px 18px; border-radius: 30px;
    font-size: 12px; font-weight: 700; cursor: pointer;
    font-family: inherit; letter-spacing: .5px;
    box-shadow:
      0 8px 14px -6px var(--clay-clear-shadow),
      inset 0 -3px 5px var(--clay-clear-inset),
      inset 0 3px 6px var(--clay-shadow-light);
    transition: all .12s ease, background .35s ease, color .35s ease;
  }
  .clear-btn:hover {
    background: var(--clay-clear-hover);
    transform: translateY(-1px);
    box-shadow:
      0 12px 18px -8px var(--clay-clear-shadow),
      inset 0 -3px 5px var(--clay-clear-inset),
      inset 0 3px 8px var(--clay-shadow-light);
  }
  .clear-btn:active {
    transform: translateY(3px);
    box-shadow:
      0 4px 8px -4px var(--clay-clear-shadow),
      inset 0 -1px 3px var(--clay-clear-inset),
      inset 0 4px 8px var(--clay-inset-g);
  }
  .file-list { display: flex; flex-direction: column; gap: 8px; max-height: 220px; overflow-y: auto; }
  .file-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px; border-radius: var(--radius-xs);
    background: var(--clay-surface);
    font-size: 13px;
    box-shadow:
      0 6px 10px -6px var(--clay-shadow-dark),
      inset 0 -3px 5px var(--clay-inset-c),
      inset 0 3px 6px var(--clay-inset-g);
    border: 1px solid var(--clay-inset-e);
    transition: background .35s ease, border-color .35s ease;
  }
  .file-name {
    color: var(--clay-text); font-family: 'JetBrains Mono', monospace;
    font-size: 12px; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; flex: 1; font-weight: 500;
    transition: color .35s ease;
  }
  .file-size { color: var(--clay-text-soft); font-size: 12px; margin-left: 12px; font-variant-numeric: tabular-nums; font-weight: 600; transition: color .35s ease; }

  /* ───────── CLAY PANEL ───────── */
  .panel {
    background: var(--clay-surface-2);
    border: 1px solid var(--clay-inset-f);
    border-radius: var(--radius-sm);
    padding: 28px;
    margin-bottom: 26px;
    box-shadow:
      0 22px 32px -18px var(--clay-shadow-dark),
      inset 0 -8px 14px var(--clay-inset-b),
      inset 0 8px 16px var(--clay-shadow-light);
    transition: background .35s ease, border-color .35s ease;
  }
  .panel-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 24px;
  }
  .panel-header h2 {
    font-size: 18px; font-weight: 700; letter-spacing: -0.3px;
    color: var(--clay-text);
    text-shadow: 0 1px 2px var(--clay-shadow-light);
    transition: color .35s ease;
  }
  .pill-sm {
    font-size: 11px; font-weight: 700; padding: 6px 14px;
    border-radius: 30px;
    background: var(--clay-surface-3);
    color: var(--clay-text-soft);
    border: 1px solid var(--clay-inset-f);
    box-shadow:
      0 4px 8px -4px var(--clay-shadow-dark),
      inset 0 -2px 4px var(--clay-inset-c),
      inset 0 2px 4px var(--clay-inset-g);
    transition: all .35s ease;
  }
  .pill-sm.success {
    color: var(--clay-success-text);
    background: var(--clay-success-bg);
    border-color: var(--clay-success-border);
  }

  /* ───────── SETTINGS ROWS ───────── */
  .setting {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 18px; align-items: center;
    padding: 18px 0;
    border-bottom: 1px solid var(--clay-divider);
    transition: border-color .35s ease;
  }
  .setting:last-child { border-bottom: none; }
  .setting-info label {
    display: block; font-size: 15px; font-weight: 700; margin-bottom: 3px;
    color: var(--clay-text);
    text-shadow: 0 1px 1px var(--clay-shadow-light);
    transition: color .35s ease;
  }
  .hint { font-size: 12px; color: var(--clay-text-soft); line-height: 1.4; transition: color .35s ease; }

  .setting.target {
    background: var(--clay-surface);
    border: 1px solid var(--clay-inset-g);
    border-radius: var(--radius-xs);
    padding: 20px 22px; margin-bottom: 8px;
    box-shadow:
      0 10px 18px -10px var(--clay-shadow-dark),
      inset 0 -4px 8px var(--clay-inset-c),
      inset 0 4px 10px var(--clay-shadow-light);
    transition: background .35s ease, border-color .35s ease;
  }
  .setting.target label { font-size: 16px; font-weight: 700; }

  input[type="number"], select {
    background: var(--clay-surface);
    border: 1px solid var(--clay-inset-g);
    color: var(--clay-text);
    padding: 12px 18px;
    border-radius: var(--radius-xs);
    font: inherit; font-size: 14px; font-weight: 600;
    outline: none;
    transition: all .2s ease, background .35s ease, color .35s ease;
    box-shadow:
      0 6px 10px -6px var(--clay-shadow-dark),
      inset 0 -3px 6px var(--clay-inset-c),
      inset 0 3px 6px var(--clay-shadow-light);
  }
  input[type="number"]:focus, select:focus {
    border-color: var(--clay-dz-hover);
    box-shadow:
      0 8px 14px -6px var(--clay-shadow-dark),
      inset 0 -3px 6px var(--clay-inset-b),
      inset 0 3px 8px var(--clay-shadow-light);
  }
  input[type="number"] { width: 120px; text-align: center; }
  input[type="number"]::placeholder { color: var(--clay-placeholder); font-weight: 400; }
  .input-group { display: flex; gap: 8px; align-items: center; }
  .input-group .unit { color: var(--clay-text-soft); font-size: 13px; font-weight: 600; transition: color .35s ease; }

  /* ───────── CLAY CHIPS (preset buttons) ───────── */
  .chip-row { display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; max-width: 440px; }
  .chip {
    position: relative;
    background: var(--clay-surface);
    color: var(--clay-text-soft);
    padding: 11px 20px; border-radius: 40px;
    border: 1px solid var(--clay-inset-g);
    font-size: 13px; font-weight: 700; cursor: pointer;
    font-family: inherit; letter-spacing: .3px;
    box-shadow:
      0 10px 16px -8px var(--clay-shadow-dark),
      inset 0 -4px 6px var(--clay-inset-c),
      inset 0 4px 8px var(--clay-shadow-light);
    transition: all .12s ease, background .35s ease, color .35s ease, border-color .35s ease;
  }
  .chip:hover {
    color: var(--clay-text);
    background: var(--clay-surface-2);
    transform: translateY(-2px);
    box-shadow:
      0 14px 20px -8px var(--clay-shadow-dark),
      inset 0 -4px 6px var(--clay-inset-b),
      inset 0 4px 10px var(--clay-shadow-light);
  }
  .chip:active {
    transform: translateY(3px);
    box-shadow:
      0 4px 8px -4px var(--clay-shadow-dark),
      inset 0 -2px 4px var(--clay-inset-c),
      inset 0 4px 8px var(--clay-inset-g);
  }
  .chip.active {
    background: var(--clay-accent);
    color: var(--clay-text);
    border-color: var(--clay-active-border);
    box-shadow:
      0 12px 20px -8px var(--clay-shadow-dark),
      inset 0 -4px 8px var(--clay-inset-i),
      inset 0 4px 10px var(--clay-shadow-light);
  }
  .chip.active:hover {
    background: var(--clay-strong-bg);
    color: var(--clay-text);
  }
  .chip.active:active {
    transform: translateY(3px);
    box-shadow:
      0 6px 10px -4px var(--clay-shadow-dark),
      inset 0 -2px 4px var(--clay-inset-i),
      inset 0 4px 8px var(--clay-inset-g);
  }

  /* ───────── CLAY SLIDER ───────── */
  .slider-wrap { position: relative; width: 320px; }
  input[type="range"] {
    -webkit-appearance: none; appearance: none;
    width: 100%; height: 14px; border-radius: 20px;
    background: var(--clay-input-bg);
    outline: none; cursor: pointer;
    box-shadow:
      inset 0 4px 6px var(--clay-shadow-dark),
      inset 0 -4px 6px var(--clay-inset-g),
      0 4px 8px -4px var(--clay-shadow-dark);
    border: 1px solid var(--clay-inset-e);
    transition: background .35s ease;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 30px; height: 30px; border-radius: 50%;
    background: var(--clay-surface);
    cursor: pointer;
    box-shadow:
      0 6px 10px -4px var(--clay-shadow-dark),
      inset 0 -2px 4px var(--clay-inset-d),
      inset 0 2px 8px var(--clay-shadow-light);
    border: 1px solid var(--clay-inset-g);
    transition: transform .1s ease, background .35s ease;
  }
  input[type="range"]::-webkit-slider-thumb:hover { transform: scale(1.1); }
  input[type="range"]::-moz-range-thumb {
    width: 30px; height: 30px; border-radius: 50%; border: 1px solid var(--clay-inset-g);
    background: var(--clay-surface);
    cursor: pointer;
    box-shadow:
      0 6px 10px -4px var(--clay-shadow-dark),
      inset 0 -2px 4px var(--clay-inset-d),
      inset 0 2px 8px var(--clay-shadow-light);
  }
  .value-badge {
    display: inline-block; font-family: 'JetBrains Mono', monospace;
    font-size: 12px; padding: 3px 10px; border-radius: 20px;
    background: var(--clay-surface);
    color: var(--clay-text);
    margin-left: 8px; font-weight: 700;
    box-shadow:
      0 4px 8px -4px var(--clay-shadow-dark),
      inset 0 -2px 4px var(--clay-inset-c),
      inset 0 2px 4px var(--clay-inset-g);
    border: 1px solid var(--clay-inset-f);
    transition: background .35s ease, color .35s ease;
  }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .grid-2 .setting { grid-template-columns: 1fr auto; border-bottom: none; }

  /* ───────── CLAY TOGGLE SWITCH ───────── */
  .toggle { display: flex; align-items: center; gap: 14px; }
  .switch { position: relative; display: inline-block; width: 56px; height: 32px; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider-knob {
    position: absolute; inset: 0; cursor: pointer;
    background: var(--clay-input-bg);
    border-radius: 32px; transition: background .3s;
    box-shadow:
      inset 0 4px 6px var(--clay-shadow-dark),
      inset 0 -4px 6px var(--clay-inset-f),
      0 4px 8px -4px var(--clay-shadow-dark);
    border: 1px solid var(--clay-inset-e);
  }
  .slider-knob::before {
    content: ""; position: absolute;
    width: 24px; height: 24px; border-radius: 50%;
    background: var(--clay-surface);
    left: 4px; top: 3px;
    transition: transform .3s cubic-bezier(0.34, 1.3, 0.7, 1), background .35s ease;
    box-shadow:
      0 6px 8px -4px var(--clay-shadow-dark),
      inset 0 -2px 4px var(--clay-inset-d),
      inset 0 2px 6px var(--clay-shadow-light);
  }
  .switch input:checked + .slider-knob {
    background: var(--clay-accent-3);
  }
  .switch input:checked + .slider-knob::before {
    transform: translateX(24px);
    background: var(--clay-surface);
  }

  /* ───────── CLAY PRIMARY BUTTON ───────── */
  .btn-primary {
    position: relative;
    width: 100%;
    padding: 22px 28px;
    border: 1px solid var(--clay-inset-g);
    border-radius: var(--radius);
    background: var(--clay-accent);
    color: var(--clay-text);
    font: inherit; font-size: 17px; font-weight: 700;
    letter-spacing: 0.5px; cursor: pointer;
    text-shadow: 0 1px 2px var(--clay-shadow-light);
    box-shadow:
      0 20px 30px -14px var(--clay-shadow-dark),
      0 8px 14px -8px rgba(0, 0, 0, 0.1),
      inset 0 -6px 10px var(--clay-inset-i),
      inset 0 6px 12px var(--clay-shadow-light);
    transform: translateY(0);
    transition:
      transform .12s cubic-bezier(.2,.7,.3,1),
      box-shadow .12s cubic-bezier(.2,.7,.3,1),
      background .2s, color .35s ease, border-color .35s ease;
    overflow: hidden;
    isolation: isolate;
  }
  .btn-primary:hover:not(:disabled) {
    background: var(--clay-strong-bg);
    transform: translateY(-2px);
    box-shadow:
      0 26px 36px -16px var(--clay-shadow-dark),
      0 10px 18px -10px rgba(0, 0, 0, 0.12),
      inset 0 -6px 10px var(--clay-inset-i),
      inset 0 6px 14px var(--clay-shadow-light);
  }
  .btn-primary:active:not(:disabled) {
    transform: translateY(4px);
    box-shadow:
      0 10px 16px -10px var(--clay-shadow-dark),
      0 4px 8px -4px rgba(0, 0, 0, 0.1),
      inset 0 -3px 6px var(--clay-inset-h),
      inset 0 6px 12px var(--clay-shadow-light);
    background: var(--clay-accent-2);
  }
  .btn-primary:disabled {
    opacity: 0.5; cursor: not-allowed;
    background: var(--clay-disabled-bg);
    color: var(--clay-disabled-text);
    text-shadow: none;
    box-shadow:
      0 10px 16px -10px var(--clay-shadow-dark),
      inset 0 -4px 8px var(--clay-inset-c),
      inset 0 4px 10px var(--clay-inset-f);
    transform: translateY(0);
  }
  .btn-primary.btn-loading:disabled { opacity: 0.9; }

  /* ═══════════════════════════════════════════════════════════
     OUTER-BORDER ANIMATED LOADING EFFECT (clay variant)
     ═══════════════════════════════════════════════════════════ */
  @property --angle {
    syntax: "<angle>";
    initial-value: 0deg;
    inherits: false;
  }
  @keyframes borderRotate {
    to { --angle: 360deg; }
  }
  .btn-primary.btn-loading {
    overflow: visible;
    border-color: transparent;
    animation: borderRotate 1.6s linear infinite;
  }
  .btn-primary.btn-loading::after {
    content: "";
    position: absolute;
    inset: -5px;
    border-radius: calc(var(--radius) + 5px);
    padding: 5px;
    background: conic-gradient(
      from var(--angle),
      var(--clay-ring-1) 0deg,
      var(--clay-ring-2) 90deg,
      var(--clay-ring-3) 180deg,
      var(--clay-ring-1) 270deg,
      var(--clay-ring-2) 360deg
    );
    -webkit-mask:
      linear-gradient(#000 0 0) content-box,
      linear-gradient(#000 0 0);
    -webkit-mask-composite: xor;
            mask-composite: exclude;
    animation: borderRotate 1.6s linear infinite;
    pointer-events: none;
    z-index: 3;
  }
  @media (prefers-reduced-motion: reduce) {
    .btn-primary.btn-loading,
    .btn-primary.btn-loading::after {
      animation: none !important;
    }
  }

  /* ───────── CLAY TABS ───────── */
  .tabs {
    display: flex; gap: 10px; margin-bottom: 28px;
    background: var(--clay-surface-2);
    padding: 10px; border-radius: var(--radius-sm);
    box-shadow:
      0 16px 24px -14px var(--clay-shadow-dark),
      inset 0 -6px 10px var(--clay-inset-b),
      inset 0 6px 12px var(--clay-shadow-light);
    border: 1px solid var(--clay-inset-f);
    transition: background .35s ease, border-color .35s ease;
  }
  .tab {
    position: relative;
    flex: 1; padding: 15px 20px; border-radius: var(--radius-xs);
    background: var(--clay-surface);
    color: var(--clay-text-soft);
    border: 1px solid var(--clay-inset-g);
    font: inherit; font-weight: 700;
    font-size: 14px; cursor: pointer;
    letter-spacing: .3px;
    box-shadow:
      0 8px 14px -8px var(--clay-shadow-dark),
      inset 0 -4px 6px var(--clay-inset-c),
      inset 0 4px 8px var(--clay-shadow-light);
    transition: all .12s ease, background .35s ease, color .35s ease;
  }
  .tab:hover { color: var(--clay-text); transform: translateY(-1px); }
  .tab:active {
    transform: translateY(3px);
    box-shadow:
      0 4px 8px -4px var(--clay-shadow-dark),
      inset 0 -2px 4px var(--clay-inset-c),
      inset 0 4px 8px var(--clay-inset-f);
  }
  .tab.active {
    background: var(--clay-accent);
    color: var(--clay-text);
    border-color: var(--clay-active-border);
    box-shadow:
      0 10px 16px -8px var(--clay-shadow-dark),
      inset 0 -4px 8px var(--clay-inset-i),
      inset 0 4px 10px var(--clay-shadow-light);
  }
  .tab.active:hover { background: var(--clay-strong-bg); }
  .tab.active:active {
    transform: translateY(3px);
    box-shadow:
      0 6px 10px -4px var(--clay-shadow-dark),
      inset 0 -2px 4px var(--clay-inset-i),
      inset 0 4px 8px var(--clay-inset-g);
  }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; animation: fadeUp .35s ease; }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* ───────── RESULTS ───────── */
  .results { display: none; }
  .results.active { display: block; animation: fadeUp .5s ease; }
  .result-item {
    background: var(--clay-surface);
    border: 1px solid var(--clay-inset-g);
    border-radius: var(--radius-xs);
    padding: 18px 20px;
    margin-bottom: 12px;
    box-shadow:
      0 10px 16px -10px var(--clay-shadow-dark),
      inset 0 -4px 8px var(--clay-inset-c),
      inset 0 4px 10px var(--clay-shadow-light);
    transition: background .35s ease, border-color .35s ease;
  }
  .result-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px;
  }
  .result-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px; color: var(--clay-text); font-weight: 500;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    transition: color .35s ease;
  }
  .result-badge {
    font-size: 11px; font-weight: 700;
    padding: 5px 12px; border-radius: 30px;
    background: var(--clay-success-bg);
    color: var(--clay-success-text);
    border: 1px solid var(--clay-success-border);
    flex-shrink: 0; margin-left: 10px;
    box-shadow:
      0 3px 6px -3px var(--clay-shadow-dark),
      inset 0 -2px 3px var(--clay-inset-h),
      inset 0 2px 4px var(--clay-inset-g);
    transition: all .35s ease;
  }
  .result-stats {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 12px; color: var(--clay-text-dim);
    gap: 12px; font-weight: 600;
    transition: color .35s ease;
  }

  /* ───────── CLAY DOWNLOAD BUTTON ───────── */
  .dl-btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 10px 18px; border-radius: var(--radius-xs);
    background: var(--clay-surface-2);
    color: var(--clay-text);
    border: 1px solid var(--clay-inset-g);
    font-size: 12px; font-weight: 700; text-decoration: none;
    letter-spacing: .3px; font-family: inherit;
    box-shadow:
      0 8px 14px -8px var(--clay-shadow-dark),
      inset 0 -3px 5px var(--clay-inset-h),
      inset 0 3px 6px var(--clay-shadow-light);
    transition: all .12s ease, background .35s ease, color .35s ease;
  }
  .dl-btn:hover {
    background: var(--clay-accent);
    transform: translateY(-2px);
    box-shadow:
      0 12px 18px -8px var(--clay-shadow-dark),
      inset 0 -3px 5px var(--clay-inset-i),
      inset 0 3px 8px var(--clay-shadow-light);
  }
  .dl-btn:active {
    transform: translateY(3px);
    box-shadow:
      0 4px 8px -4px var(--clay-shadow-dark),
      inset 0 -2px 3px var(--clay-inset-c),
      inset 0 3px 6px var(--clay-inset-g);
  }
  .error-item {
    background: var(--clay-error-bg);
    border: 1px solid var(--clay-error-border);
    color: var(--clay-error-text);
    padding: 14px 18px; border-radius: var(--radius-xs);
    font-size: 13px; margin-bottom: 10px; font-weight: 600;
    box-shadow:
      0 6px 10px -6px var(--clay-err-shadow),
      inset 0 -3px 5px var(--clay-err-inset-a),
      inset 0 3px 6px var(--clay-err-inset-b);
    transition: all .35s ease;
  }

  .foot {
    display: flex; justify-content: space-between;
    font-size: 12px; color: var(--clay-text-soft);
    padding-top: 40px; opacity: 0.8; font-weight: 600;
    transition: color .35s ease;
  }

  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb {
    background: var(--clay-scroll-thumb);
    border-radius: 999px;
    border: 2px solid var(--clay-bg);
    transition: background .35s ease;
  }

  @media (max-width: 720px) {
    .nav { padding: 14px 18px; flex-wrap: wrap; gap: 10px; }
    .theme-picker { order: 3; width: 100%; justify-content: center; }
    .theme-label { display: none; }
    .container { padding: 40px 16px 30px; }
    .grid-2 { grid-template-columns: 1fr; }
    .setting { grid-template-columns: 1fr; text-align: left; }
    .chip-row { justify-content: flex-start; }
    .slider-wrap { width: 100%; }
    .panel { padding: 20px; }
    .dropzone { padding: 40px 20px; }
  }
</style>
</head>
<body>
  <!-- floating clay pebbles -->
  <div class="pebble p1"></div>
  <div class="pebble p2"></div>
  <div class="pebble p3"></div>
  <div class="pebble p4"></div>
  <div class="pebble p5"></div>

  <nav class="nav">
    <div class="brand">
      
      <span>Image<strong>.Compressor</strong></span>
    </div>
    <!-- ═══════════ COLOUR THEME PICKER ═══════════ -->
    <div class="theme-picker">
      <span class="theme-label">Theme</span>
      <div class="theme-swatches" id="themeSwatches">
        <button class="theme-swatch sw-beige active" data-theme="beige" title="Beige" aria-label="Beige theme"></button>
        <button class="theme-swatch sw-mint" data-theme="mint" title="Mint" aria-label="Mint theme"></button>
        <button class="theme-swatch sw-lavender" data-theme="lavender" title="Lavender" aria-label="Lavender theme"></button>
        <button class="theme-swatch sw-peach" data-theme="peach" title="Peach" aria-label="Peach theme"></button>
        <button class="theme-swatch sw-sky" data-theme="sky" title="Sky" aria-label="Sky theme"></button>
        <button class="theme-swatch sw-rose" data-theme="rose" title="Rose" aria-label="Rose theme"></button>
        <button class="theme-swatch sw-charcoal" data-theme="charcoal" title="Charcoal" aria-label="Charcoal theme"></button>
      </div>
    </div>
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
            <label>Target file size</label>
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
            <h2>Results</h2>
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

  // ═══════════════════════════════════════════════════════════
  //  COLOUR THEME SWITCHER
  //  Applies `data-theme` on <html>, saves to localStorage.
  // ═══════════════════════════════════════════════════════════
  const THEME_KEY = 'clay-theme';
  const swatches = document.querySelectorAll('.theme-swatch');

  function applyTheme(name) {
    const valid = ['beige','mint','lavender','peach','sky','rose','charcoal'];
    if (!valid.includes(name)) name = 'beige';
    document.documentElement.setAttribute('data-theme', name);
    swatches.forEach(sw => sw.classList.toggle('active', sw.dataset.theme === name));
    try { localStorage.setItem(THEME_KEY, name); } catch (e) {}
  }

  swatches.forEach(sw => {
    sw.addEventListener('click', () => applyTheme(sw.dataset.theme));
  });

  // Restore saved theme on load
  try {
    const saved = localStorage.getItem(THEME_KEY);
    applyTheme(saved || 'beige');
  } catch (e) {
    applyTheme('beige');
  }

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
  //  TAB 1: IMAGE COMPRESSOR (unchanged logic)
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
    compressBtn.classList.add('btn-loading');
    compressBtn.textContent = 'Compressing…';
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
    compressBtn.classList.remove('btn-loading');
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
  //  TAB 2: PDF → WORD (unchanged logic)
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

  const engineChips = document.querySelectorAll('#engineChips .chip');
  let activeEngine = 'pdf2docx';

  engineChips.forEach(chip => {
    chip.addEventListener('click', () => {
      engineChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      activeEngine = chip.dataset.engine;
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
    pdfConvertBtn.classList.add('btn-loading');
    pdfConvertBtn.textContent =
      `Converting via ${activeEngine === 'libreoffice' ? 'LibreOffice' : 'pdf2docx'}…`;
    pdfResultsEl.classList.remove('active');
    pdfResultListEl.innerHTML = '';

    const allResults = [];
    for (const f of pdfFiles) {
      const form = new FormData();
      form.append('file', f);
      form.append('engine', activeEngine);
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
    pdfConvertBtn.classList.remove('btn-loading');
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