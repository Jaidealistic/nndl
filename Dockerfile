FROM python:3.11-slim

# System dependencies for PyMuPDF, OpenCV, and font rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install CPU-optimized PyTorch first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py index.html /app/
COPY demo_samples/ /app/demo_samples/
COPY src/ /app/src/
COPY dataset/dataset_generator.py /app/dataset/

# Create uploads directory with write permissions
RUN mkdir -p /app/uploads && chmod 777 /app/uploads

# Configure Port (Hugging Face Spaces uses 7860, Render uses 10000 or $PORT)
ENV PORT=7860
EXPOSE 7860

# Start production server with Gunicorn
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 2 --timeout 120 app:app
