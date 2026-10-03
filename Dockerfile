FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by PyMuPDF / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model during build for instant cold-starts
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application source code
COPY . .

# Create writable permissions for SQLite checkpointer and Hugging Face Spaces non-root user
RUN chmod -R 777 /app

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api:api --host 0.0.0.0 --port ${PORT:-7860}"]
