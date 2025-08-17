# Use a lightweight Python image
FROM python:3.11.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for numpy, chromadb, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

# Expose Hugging Face Spaces port
EXPOSE 7860

# Run Flask app on port 7860
CMD ["python", "app.py"]
