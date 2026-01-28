import io
import os
import math
import subprocess
from typing import Tuple, Optional

import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)

# Limit uploads (bytes). Default 25MB, adjust as needed.
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

# Model choices (rembg CLI -m values)
MODEL_PRIMARY_PHOTO = os.getenv("MODEL_PRIMARY_PHOTO", "bria-rmbg")
MODEL_FALLBACK_PHOTO = os.getenv("MODEL_FALLBACK_PHOTO", "birefnet-general")

MODEL_PRIMARY_LOGO = os.getenv("MODEL_PRIMARY_LOGO", "isnet-general-use")
MODEL_FALLBACK_LOGO = os.getenv("MODEL_FALLBACK_LOGO", "bria-rmbg")

# Speed/quality knobs
PHOTO_MAX_SIDE_FOR_MODEL = int(os.getenv("PHOTO_MAX_SIDE_FOR_MODEL", "1024"))
LOGO_MAX_SIDE_FOR_MODEL = int(os.getenv("LOGO_MAX_SIDE_FOR_MODEL", "1536"))
REMBG_TIMEOUT_SEC = int(os.getenv("REMBG_TIMEOUT_SEC", "120"))

# Heuristic thresholds
LOGO_UNIQUE_COLORS_MAX = int(os.getenv("LOGO_UNIQUE_COLORS_MAX", "180"))
LOGO_ENTROPY_MAX = float(os.getenv("LOGO_ENTROPY_MAX", "5.2"))
LOGO_EDGE_DENSITY_MIN = float(os.getenv("LOGO_EDGE_DENSITY_MIN", "0.08"))


def pil_open_safely(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im.load()
    return im


def to_rgba(im: Image.Image) -> Image.Image:
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    return im


def resize_max_side(im: Image.Image, max_side: int) -> Image.Image:
    w, h = im.size
    if max(w, h) <= max_side:
        return im
    scale = max_side / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return im.resize((new_w, new_h), Image.LANCZOS)


def shannon_entropy(gray_u8: np.ndarray) -> float:
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    p = hist / max(1.0, hist.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def estimate_logo_like(im_rgba: Image.Image) -> bool:
    # Downsample for fast stats
    small = im_rgba.convert("RGB").resize((256, 256), Image.BILINEAR)
    arr = np.array(small)

    # Unique colors (after quantizing to reduce noise)
    # (Quantize by bucketing each channel to 32 levels)
    q = (arr // 8).astype(np.uint8)
    uniq_colors = np.unique(q.reshape(-1, 3), axis=0).shape[0]

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    ent = shannon_entropy(gray)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())

    # Logo-like: fewer colors, lower entropy, more “graphic” edges
    score = 0
    if uniq_colors <= LOGO_UNIQUE_COLORS_MAX:
        score += 1
    if ent <= LOGO_ENTROPY_MAX:
        score += 1
    if edge_density >= LOGO_EDGE_DENSITY_MIN:
        score += 1

    return score >= 2


def rembg_cli_remove(input_bytes: bytes, model: str, alpha_matting: bool) -> bytes:
    """
    Uses rembg CLI reading from stdin and writing to stdout.
    Supported flags shown in rembg docs: -m (model), -a (alpha matting). :contentReference[oaicite:3]{index=3}
    """
    cmd = ["rembg", "i", "-m", model]
    if alpha_matting:
        cmd.append("-a")

    try:
        p = subprocess.run(
            cmd,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=REMBG_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"rembg timed out after {REMBG_TIMEOUT_SEC}s")

    if p.returncode != 0 or not p.stdout:
        err = (p.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"rembg failed (model={model}). stderr: {err[:2000]}")

    return p.stdout


def alpha_quality_score(alpha: np.ndarray) -> float:
    """
    Quick sanity: if alpha is nearly all 0 or all 255, it likely failed.
    Returns a score where higher is “more plausible”.
    """
    a = alpha.astype(np.float32) / 255.0
    mean = float(a.mean())
    var = float(a.var())
    # Encourage “some background removed” and “some structure”
    score = (1.0 - abs(mean - 0.65)) + min(0.5, var * 2.0)
    return score


def extract_alpha(png_bytes: bytes) -> Tuple[Image.Image, np.ndarray]:
    out = pil_open_safely(png_bytes)
    out = to_rgba(out)
    a = np.array(out)[:, :, 3]
    return out, a


def refine_alpha_for_logo(alpha_u8: np.ndarray) -> np.ndarray:
    """
    Logos want crisp edges.
    - remove tiny specks
    - slightly harden edge
    """
    a = alpha_u8.copy()

    # Clean isolated noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, k, iterations=1)
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, k, iterations=1)

    # Harden edge a bit (gamma)
    af = (a.astype(np.float32) / 255.0)
    af = np.clip(af, 0, 1) ** 1.15
    a = np.clip(af * 255.0, 0, 255).astype(np.uint8)

    return a


def refine_alpha_for_photo(alpha_u8: np.ndarray) -> np.ndarray:
    """
    Photos want smoother edges (hair/soft boundaries).
    - gentle blur on the transition band
    - avoid losing fine details
    """
    a = alpha_u8.copy()
    # Mild bilateral smoothing on alpha to reduce jaggies without destroying edges
    a = cv2.bilateralFilter(a, d=5, sigmaColor=25, sigmaSpace=25)
    return a


def estimate_solid_bg_color(rgb: np.ndarray, alpha: np.ndarray) -> Optional[np.ndarray]:
    """
    If background is mostly solid, estimate bg color from sure-background pixels.
    Returns BGR or None.
    """
    bg = rgb[alpha < 10]
    if bg.size < 500:  # not enough bg pixels
        return None

    # If bg is not “close to solid”, skip
    std = bg.reshape(-1, 3).std(axis=0).mean()
    if std > 18.0:
        return None

    med = np.median(bg.reshape(-1, 3), axis=0)
    return med.astype(np.float32)


def decontaminate_edges(rgba: np.ndarray) -> np.ndarray:
    """
    Reduce halo/fringing when background was a solid color.
    This is a simplified “color decontamination” step (helps logos a lot).
    """
    rgb = rgba[:, :, :3].astype(np.float32)
    a = rgba[:, :, 3].astype(np.float32) / 255.0

    bg = estimate_solid_bg_color(rgb, (a * 255).astype(np.uint8))
    if bg is None:
        return rgba

    # Un-premultiply against estimated bg
    # rgb = fg*a + bg*(1-a)  =>  fg = (rgb - bg*(1-a)) / max(a, eps)
    eps = 1e-4
    fg = (rgb - bg[None, None, :] * (1.0 - a[..., None])) / np.maximum(a[..., None], eps)
    fg = np.clip(fg, 0, 255).astype(np.uint8)

    out = rgba.copy()
    out[:, :, :3] = fg
    return out


def apply_alpha_to_original(original_rgba: Image.Image, alpha_small: np.ndarray, small_size: Tuple[int, int]) -> bytes:
    """
    We run the model on a smaller image, then upscale alpha to original size
    and apply it to the original RGB. This keeps output high-res and faster.
    """
    ow, oh = original_rgba.size
    sw, sh = small_size

    # Choose interpolation: photos benefit from smoother alpha, logos from sharper alpha
    # We'll decide outside and pass in refined alpha already.
    alpha_up = cv2.resize(alpha_small, (ow, oh), interpolation=cv2.INTER_LANCZOS4)

    orig_arr = np.array(original_rgba)
    orig_arr[:, :, 3] = alpha_up.astype(np.uint8)

    # Optional halo cleanup (only helps when bg is solid)
    orig_arr = decontaminate_edges(orig_arr)

    out = Image.fromarray(orig_arr, mode="RGBA")
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def get_uploaded_file():
    # Accept both common keys and fallback to "first file"
    f = request.files.get("image") or request.files.get("file")
    if not f and request.files:
        f = next(iter(request.files.values()))
    return f

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "primary_photo_model": MODEL_PRIMARY_PHOTO,
        "fallback_photo_model": MODEL_FALLBACK_PHOTO,
        "primary_logo_model": MODEL_PRIMARY_LOGO,
        "fallback_logo_model": MODEL_FALLBACK_LOGO,
    })


@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    try:
        f = get_uploaded_file()
        if not f:
            return jsonify({
                "error": "No file received",
                "content_type": request.content_type,
                "files_keys": list(request.files.keys()),
                "form_keys": list(request.form.keys()),
            }), 400

        data = f.read()
        if not data:
            return jsonify({"error": "Empty file received"}), 400

        # TODO: run your rembg/PIL pipeline using `data`
        # result_bytes = ...

        # return send_file(BytesIO(result_bytes), mimetype="image/png")
        return jsonify({"ok": True, "filename": f.filename, "size": len(data)})

    except Exception as e:
        return jsonify({
            "error": "Server failed processing",
            "detail": str(e),
            "content_type": request.content_type,
            "files_keys": list(request.files.keys()),
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
