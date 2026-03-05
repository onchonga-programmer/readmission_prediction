FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies first (Docker caches this layer)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the script
COPY ingest.py .

# Run the script when the container starts
CMD ["python", "ingest.py"]