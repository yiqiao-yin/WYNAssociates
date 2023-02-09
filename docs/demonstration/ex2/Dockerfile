# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set the environment variable for the API key
ENV API_KEY=1617249c979543caaef58e3c726b7ff3

# Install the curl tool
RUN apt-get update && apt-get install -y curl

# Define the command to run when the container starts
RUN curl https://api.synthesia.io/v2/videos  \
  -H "Authorization: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{ "test": true, "input": [{ "scriptText": "Hello, World! This is my first synthetic video, made with the Synthesia API!", "avatar": "anna_costume1_cameraA", "background": "green_screen"}] }'