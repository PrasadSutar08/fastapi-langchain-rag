FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Controls whether the heavy self-hosted-embedding stack (torch,
# sentence-transformers, langchain-huggingface) gets installed at all.
# Default true: local `docker compose build` and anyone running this
# project open-source gets EMBEDDING_PROVIDER=local working out of the
# box, matching this project's default settings.
#
# The deployed image is built with --build-arg INSTALL_LOCAL_EMBEDDINGS=false
# alongside EMBEDDING_PROVIDER=cohere in its environment -- skipping this
# install is what actually makes the image fit in a free host's ~512MB
# RAM/build-memory limit; toggling EMBEDDING_PROVIDER alone doesn't help
# if these packages are still being installed underneath it.
ARG INSTALL_LOCAL_EMBEDDINGS=true

COPY requirements.txt requirements-local-embeddings.txt /

# Install CPU-only torch FIRST, from PyTorch's own CPU wheel index,
# when local embeddings are enabled. This app only ever runs the
# embedding model on CPU (see EMBEDDING_DEVICE=cpu) -- Ollama runs as
# a separate process/service and never touches this container's torch
# install. The default PyPI torch wheel bundles ~2.5GB of unused
# NVIDIA CUDA libraries; the CPU wheel is a fraction of that.
# Installing it first means the later requirements-local-embeddings.txt
# install sees torch==2.5.1 already satisfied and skips redownloading it.
RUN if [ "$INSTALL_LOCAL_EMBEDDINGS" = "true" ]; then \
        pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
        pip install --no-cache-dir -r /requirements-local-embeddings.txt ; \
    fi

RUN pip install --no-cache-dir -r /requirements.txt

COPY backend /app/backend
WORKDIR /app/backend

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]