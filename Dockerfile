# Use the official Python image from the Docker Hub
FROM python:3.11-slim


# HuggingFace User needed!
# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# # Set home to the user's home directory
# ENV HOME=/home/user \
# 	PATH=/home/user/.local/bin:$PATH

# # Set the working directory to the user's home directory
WORKDIR /app

# Try and run pip command after setting the user with `USER user` to avoid permission issues with Python
RUN pip install --no-cache-dir --upgrade pip

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
# COPY --chown=user . $HOME/app

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
