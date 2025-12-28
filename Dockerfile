FROM python:3.11-slim

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết 
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng Docker Cache
COPY requirements.txt .

# Cài đặt Python serve dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code backend vào
COPY . .

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Lệnh chạy default
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
