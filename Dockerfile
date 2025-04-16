# Use an official Python runtime as a parent image
FROM python:3.12

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y libgl1

# Set the working directory in the container
WORKDIR /app

# Copy the entire project into the container's /app directory
COPY . /app

# Install the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV FLASK_APP=run.py

# Expose port 5000 to the outside world
EXPOSE 5000

# Run the Flask application when the container starts
CMD ["python", "run.py"]