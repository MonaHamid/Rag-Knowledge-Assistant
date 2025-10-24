FROM python:3.10-slim

WORKDIR /app

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy deps separately for caching
COPY requirements.txt .

# Pre-install pip and limit cache size
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt --timeout=1200

# Copy rest of app
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
