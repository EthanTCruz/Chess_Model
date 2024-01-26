# Use an official Python runtime as a parent image
FROM python:3.11.0

# Set the working directory in the container
WORKDIR /usr/src/app

RUN mkdir  /usr/src/app/Chess_Model

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app/Chess_Model

# Update the package list and install neovim
RUN apt-get update && \
    apt-get install -y neovim tmux && \
    rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /usr/src/app/Chess_Model/requirements.txt

# Define environment variable
ENV trainModel False
ENV selfTrain True
ENV GOOGLE_APPLICATION_CREDENTIALS '/var/secrets/google/key.json'
ENV BUCKET_NAME "chess-model-weights"

#CMD ["python","./Chess_Model/src/model/main.py"]
CMD while true; do sleep 10; done