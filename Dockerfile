FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf OLLAMA_HOST=http://host.docker.internal:11434

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY app.py .
COPY eval/results/ eval/results/

# Source PDFs are downloaded from the official publishers at build time (not redistributed),
# then chunked and the embedding/reranker models are pre-fetched so the container starts warm.
ARG PREBUILD=1
RUN if [ "$PREBUILD" = "1" ]; then python scripts/download_corpora.py && python scripts/build_index.py; fi

RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "ragkit.api:app", "--host", "0.0.0.0", "--port", "8000"]
