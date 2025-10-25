FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    pkg-config \
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf2.0-dev \
    libffi-dev \
    shared-mime-info \
    libxml2-dev \
    libxslt-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    fonts-dejavu \
    fonts-liberation \
    fonts-freefont-ttf \
    fonts-roboto \
    fonts-noto \
    fonts-ubuntu \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user (optional but recommended)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Start gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]