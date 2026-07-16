# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# requirements-lock.txt is the source of truth for the build — every package
# is pinned to one exact version (frozen from the first successful build),
# so pip never has to resolve/backtrack, it just installs what's listed.
# Only edit requirements.txt directly when deliberately upgrading something.
COPY requirements-lock.txt .

# --mount=type=cache keeps pip's download cache between builds, on your local
# Docker daemon, even across "docker compose build". As long as this layer's
# inputs (this COPY + the pip install command) don't change, Docker also just
# reuses the whole layer instantly instead of re-running pip at all.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-lock.txt

# Pre-download NLTK data at build time. unstructured's docx/pdf partitioners
# call nltk.sent_tokenize() for paragraph classification; without this, the
# first call tries to download punkt at runtime and fails with HTTP 403 in
# restricted network environments.
RUN python -m nltk.downloader -d /usr/local/share/nltk_data \
    punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng

# Pre-download unstructured's YOLOX layout-detection model (~217MB) at build
# time, the same way — otherwise the first real load_pdf(strategy="hi_res")
# call at runtime pauses for 1-2 minutes to fetch it, and since it's cached
# in the container's writable layer (not the image), it re-downloads on
# every container rebuild. Deliberately placed BEFORE `COPY app ./app`: it
# only depends on installed packages, not application code, so editing your
# own app/ files later never invalidates this layer or re-triggers the
# download. Builds its own throwaway PDF with pypdf (already installed)
# rather than needing a real sample file in the build context.
# IMPORTANT: no --mount=type=cache here (unlike the pip steps above) —
# a cache mount's contents are discarded when the layer is committed, so
# the downloaded model would never actually end up in the final image.
RUN python -c "\
from pypdf import PdfWriter; \
w = PdfWriter(); w.add_blank_page(width=200, height=200); \
w.write('/tmp/warmup.pdf'); \
import pdfminer.pdfparser as p; \
from pdfminer.psparser import PSSyntaxError as E; \
(not hasattr(p, 'PSSyntaxError')) and setattr(p, 'PSSyntaxError', E); \
from unstructured.partition.pdf import partition_pdf; \
partition_pdf(filename='/tmp/warmup.pdf', strategy='hi_res'); \
print('YOLOX layout model cached at build time')"

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
