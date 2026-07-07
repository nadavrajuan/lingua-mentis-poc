FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch (server has no GPU; avoids 3GB CUDA libraries)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

RUN mkdir -p /app/data /app/models

EXPOSE 8000

CMD ["sh", "-c", "python scripts/bootstrap.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
