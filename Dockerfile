FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    TORCH_HOST="download.pytorch.org" \
    TORCH_PATH="whl/cpu"

RUN pip install --no-cache-dir --default-timeout=1000 torch --index-url "https://${TORCH_HOST}/${TORCH_PATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY src/ ./src/
COPY rag_vector_db.pkl .
COPY model_checkpoint.pth .
COPY vectorizer.pkl .
COPY label_encoder.pkl .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0')" || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
