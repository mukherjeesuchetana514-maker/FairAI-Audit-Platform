# Use official Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code (main.py, Frontend folder, etc.)
COPY . .

# Cloud Run assigns a dynamic PORT environment variable. 
# We must tell Uvicorn to listen to that specific port.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]