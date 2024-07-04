#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | sed 's/#.*//g' | xargs)
fi

# Define the image and container names
IMAGE_NAME="malware-thesis-image"
CONTAINER_NAME="malware-server-container"

# Stop and remove the existing container if it exists
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Build the Docker image
docker build -t $IMAGE_NAME .

# Run the Docker container
docker run --name $CONTAINER_NAME -p 8000:8000 -v $FILE_DIR:/app/uploads malware-thesis-server