FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Put rembg models somewhere inside the container (not just /root)
ENV U2NET_HOME=/app/.u2net

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model at build time (prevents slow/failed first request)
RUN python - <<'PY'
from rembg import new_session
# "isnet-general-use" is one of the official rembg models and is a strong general default
new_session("isnet-general-use")
print("rembg model cache warmed")
PY

COPY app.py .

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "--preload"]