# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Create a non-root user with explicit permissions
RUN addgroup --gid 1000 appuser && \
    adduser --uid 1000 --gid 1000 --disabled-password appuser && \
    mkdir -p /app/uploads && \
    chown -R appuser:appuser /app/uploads

# # Set the working directory to the user's home directory
WORKDIR /app

# Try and run pip command after setting the user with `USER user` to avoid permission issues with Python
RUN pip install --no-cache-dir --upgrade pip

# Switch to the non-root user
USER appuser

# Copy the requirements file into the image
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the image
COPY . .

# Expose the port the app runs on
EXPOSE 7860

# Set the environment variable for Flask
ENV FLASK_APP=app

# Command to run the Flask app
CMD ["python", "src/app.py"]
