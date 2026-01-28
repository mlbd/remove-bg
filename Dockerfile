FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# IMPORTANT: rembg downloads models to ~/.u2net/
# We set HOME to a folder you can mount as a persistent volume in Coolify.
ENV HOME=/models
RUN mkdir -p /models

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

EXPOSE 5000

# Configurable via env
ENV GUNICORN_WORKERS=1
ENV GUNICORN_TIMEOUT=180

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:5000 --workers ${GUNICORN_WORKERS} --timeout ${GUNICORN_TIMEOUT}"]
