import os
import logging
from io import BytesIO
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from PIL import Image, UnidentifiedImageError

from rembg import remove, new_session


# -----------------------------
# App Factory
# -----------------------------
def create_app():
    app = Flask(__name__)

    # Max upload size (default: 25MB)
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024

    # Logging
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    app.logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

    # Reuse rembg model session (big performance win)
    model_name = os.getenv("REMBG_MODEL", "u2net")  # u2net, u2netp, isnet-general-use, ...
    session = new_session(model_name)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _get_uploaded_file():
        """
        Finds the uploaded file from multipart/form-data.
        Accepts common keys: 'image' or 'file', otherwise picks the first file.
        """
        f = request.files.get("image") or request.files.get("file")
        if not f and request.files:
            f = next(iter(request.files.values()))
        return f

    def _read_upload_bytes():
        f = _get_uploaded_file()
        if not f:
            return None, None, ("No file received. Send multipart/form-data with key 'image'.", 400)

        data = f.read()
        if not data:
            return None, None, ("Empty file received.", 400)

        filename = f.filename or "upload"
        return data, filename, None

    def _ensure_image_bytes(image_bytes: bytes) -> Image.Image:
        """
        Validates and returns a PIL Image from bytes.
        """
        try:
            img = Image.open(BytesIO(image_bytes))
            img.load()
            return img
        except UnidentifiedImageError:
            raise ValueError("Uploaded file is not a valid image.")

    def _to_png_bytes(img: Image.Image) -> bytes:
        out = BytesIO()
        img.save(out, format="PNG", optimize=True)
        out.seek(0)
        return out.getvalue()

    def _trim_transparent(img: Image.Image, pad: int = 0) -> Image.Image:
        """
        Crops extra transparent area from an RGBA image.
        """
        rgba = img.convert("RGBA")
        alpha = rgba.split()[-1]
        bbox = alpha.getbbox()  # bounding box of non-zero alpha

        # If fully transparent or no bbox, return as-is
        if not bbox:
            return rgba

        left, top, right, bottom = bbox
        if pad > 0:
            left = max(0, left - pad)
            top = max(0, top - pad)
            right = min(rgba.width, right + pad)
            bottom = min(rgba.height, bottom + pad)

        return rgba.crop((left, top, right, bottom))

    def _safe_output_name(original_name: str, suffix: str) -> str:
        stem = Path(original_name).stem or "output"
        return f"{stem}{suffix}.png"

    # -----------------------------
    # Routes
    # -----------------------------
    @app.get("/health")
    def health():
        return jsonify({"ok": True}), 200

    @app.post("/remove-bg")
    def remove_bg():
        """
        Upload an image (multipart/form-data) and get PNG with transparent background.
        """
        try:
            data, filename, err = _read_upload_bytes()
            if err:
                msg, code = err
                return jsonify({"error": msg}), code

            # Validate image early (better error messages)
            _ = _ensure_image_bytes(data)

            # Optional alpha matting (slower but better edges sometimes)
            alpha_matting = os.getenv("ALPHA_MATTING", "0") == "1"
            if alpha_matting:
                out_bytes = remove(
                    data,
                    session=session,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=int(os.getenv("AM_FG", "240")),
                    alpha_matting_background_threshold=int(os.getenv("AM_BG", "10")),
                    alpha_matting_erode_size=int(os.getenv("AM_ERODE", "10")),
                )
            else:
                out_bytes = remove(data, session=session)

            out_name = _safe_output_name(filename, "-nobg")

            return send_file(
                BytesIO(out_bytes),
                mimetype="image/png",
                as_attachment=False,
                download_name=out_name,
            )

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            app.logger.exception("remove-bg failed")
            return jsonify({"error": "Failed to process image", "detail": str(e)}), 500

    @app.post("/trim-transparent")
    def trim_transparent():
        """
        Upload an image (preferably PNG with alpha) and auto-crop extra transparent space.
        """
        try:
            data, filename, err = _read_upload_bytes()
            if err:
                msg, code = err
                return jsonify({"error": msg}), code

            img = _ensure_image_bytes(data).convert("RGBA")
            pad = int(request.args.get("pad", "0"))  # optional padding
            trimmed = _trim_transparent(img, pad=pad)

            out_bytes = _to_png_bytes(trimmed)
            out_name = _safe_output_name(filename, "-trimmed")

            return send_file(
                BytesIO(out_bytes),
                mimetype="image/png",
                as_attachment=False,
                download_name=out_name,
            )

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            app.logger.exception("trim-transparent failed")
            return jsonify({"error": "Failed to process image", "detail": str(e)}), 500

    @app.post("/remove-bg-trim")
    def remove_bg_trim():
        """
        Remove background + trim transparent space in one call.
        """
        try:
            data, filename, err = _read_upload_bytes()
            if err:
                msg, code = err
                return jsonify({"error": msg}), code

            _ = _ensure_image_bytes(data)

            out_bytes = remove(data, session=session)
            img = _ensure_image_bytes(out_bytes).convert("RGBA")

            pad = int(request.args.get("pad", "0"))
            trimmed = _trim_transparent(img, pad=pad)

            final_bytes = _to_png_bytes(trimmed)
            out_name = _safe_output_name(filename, "-nobg-trim")

            return send_file(
                BytesIO(final_bytes),
                mimetype="image/png",
                as_attachment=False,
                download_name=out_name,
            )

        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            app.logger.exception("remove-bg-trim failed")
            return jsonify({"error": "Failed to process image", "detail": str(e)}), 500

    return app


app = create_app()

if __name__ == "__main__":
    # For local testing only. In production you already use gunicorn.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
