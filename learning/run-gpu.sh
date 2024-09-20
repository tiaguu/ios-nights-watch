#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | sed 's/#.*//g' | xargs)
fi

# Define the image and container names
IMAGE_NAME="learning-gpu-image"
CONTAINER_NAME="learning-gpu-container"
DOCKERFILE_NAME="Dockerfile.gpu"

# Stop and remove the existing container if it exists
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME -f $DOCKERFILE_NAME . || { echo "Failed to build Docker image"; exit 1; }

# Run the Docker container with GPU support
echo "Running Docker container with GPU support..."
docker run -d \
  --name $CONTAINER_NAME \
  --gpus all \  # Ensure that GPU is available
  -e LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH \  # Set the environment variable
  $IMAGE_NAME || { echo "Failed to start Docker container"; exit 1; }

# Show logs
echo "Docker container is running. Showing logs..."
docker logs -f $CONTAINER_NAME