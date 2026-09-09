FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt /requirements.txt

# Install CPU-only torch FIRST, from PyTorch's own CPU wheel index.
# This app only ever runs the embedding model on CPU (see
# EMBEDDING_DEVICE=cpu) -- Ollama runs as a separate process/service
# and never touches this container's torch install. The default PyPI
# torch wheel bundles ~2.5GB of unused NVIDIA CUDA libraries; the CPU
# wheel is a fraction of that. Installing it first means the later
# `pip install -r requirements.txt` sees torch==2.5.1 already
# satisfied and skips reinstalling/redownloading it.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r /requirements.txt

COPY backend /app/backend
WORKDIR /app/backend

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]