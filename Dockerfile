FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    UV_LINK_MODE=copy \
    PATH=/app/.venv/bin:/root/.local/bin:$PATH

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock
RUN uv sync --frozen
RUN find /app/.venv/lib -path '*/site-packages/torch/lib/*.so' -exec patchelf --clear-execstack {} \;

COPY replication_infer.py /app/replication_infer.py
COPY download_models.py /app/download_models.py
RUN python /app/download_models.py --checkpoint-dir /app/outputs --allennlp-dir /app/Model

RUN python - <<'PY'
from transformers import BertConfig, BertTokenizer
BertConfig.from_pretrained("bert-base-uncased")
BertTokenizer.from_pretrained("bert-base-uncased")
PY

ENTRYPOINT ["python", "/app/replication_infer.py"]
