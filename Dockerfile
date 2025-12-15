# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
# Set the working directory in the container
WORKDIR /app

# Ensure output is sent directly to terminal (no buffering)
ENV PYTHONUNBUFFERED=1

# Install system dependencies (needed for pyaudio and other tools)
# portaudio19-dev is required for pyaudio
RUN apt-get update && apt-get install -y \
    gcc \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src ./src

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
