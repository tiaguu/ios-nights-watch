#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | sed 's/#.*//g' | xargs)
fi

# Define the image and container names
IMAGE_NAME="learning-image"
CONTAINER_NAME="learning-container"

# Stop and remove the existing container if it exists
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Build the Docker image
docker build -t $IMAGE_NAME .

# Run the Docker container
docker run -d --name $CONTAINER_NAME -v $MALWARE_DIR:/app/malware -v $GOODWARE_DIR:/app/goodware -v $GOODWARE_OPCODES_DIR:/app/goodware-opcodes -v $MALWARE_OPCODES_DIR:/app/malware-opcodes $IMAGE_NAME
