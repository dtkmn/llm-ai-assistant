# Use a current Python runtime with security updates.
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file into the image
COPY requirements.txt .

# Install the dependencies as root first
RUN pip install -r requirements.txt

# Copy the rest of the application code into the image
COPY . .

# Create a non-root user and set permissions
RUN addgroup --gid 1000 appuser && \
    adduser --uid 1000 --gid 1000 --disabled-password appuser && \
    mkdir -p /app/uploads /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /app /home/appuser/.cache

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 7860

# Set environment variables for the application
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV HF_HOME="/home/appuser/.cache/huggingface"
ENV FAST_MODE="true"
ENV APP_DEBUG="false"

# Command to run the Gradio app
CMD ["python", "src/app.py"]
