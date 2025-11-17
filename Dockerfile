# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into container
COPY . .

# Expose port if your app uses one (adjust if needed)
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command to run training
CMD ["python", "src/train.py"]