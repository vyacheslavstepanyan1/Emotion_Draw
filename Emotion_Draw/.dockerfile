# Use the specific Python version from the Docker Hub
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /Emotion_draw

# Copy the requirements.txt file into the container
COPY ./api/requirements.txt .

# Install the dependencies specified in the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container, excluding paths in .dockerignore
COPY . .

# Make port 8000 available for the app
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

RUN cd api

# Define the command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]