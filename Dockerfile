# Use the official Python 3.12 image as the base image.
FROM python:3.12

# Install system dependencies required by OpenCV (libGL)
RUN apt-get update && apt-get install -y libgl1

# Set the working directory in the container.
WORKDIR /app

# Copy all project files into the container (adjust .dockerignore if necessary).
COPY . /app

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable to specify the Flask entry point.
ENV FLASK_APP=run.py

# Expose port 5000 so external connections can reach the application.
EXPOSE 5000

# Run the Flask app.
CMD ["python", "run.py"]