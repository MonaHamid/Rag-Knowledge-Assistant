# ---- Base image ----
FROM python:3.10-slim

# ---- Set work directory ----
WORKDIR /app

# ---- Install minimal OS dependencies ----
# Only the essentials to compile or download wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Environment optimizations ----
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1200 \
    UV_THREADPOOL_SIZE=64 \
    STREAMLIT_SERVER_PORT=8501

# ---- Copy requirements first (for layer caching) ----
COPY requirements.txt .

# ---- Install dependencies ----
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy rest of app ----
COPY . .

# ---- Expose ports ----
# 8501 = Streamlit, 8000 = Prometheus metrics
EXPOSE 8501 8000

# ---- Run Streamlit ----
CMD ["streamlit", "run", "app_monitoring.py", "--server.port=8501", "--server.address=0.0.0.0"]