#Python base image
FROM python:3.11-slim

# Root to the directory for the DOCKER
WORKDIR /business_logic

# Requirements file
COPY requirements.txt .

# Python dependencies
# Install latest pip if needed
RUN pip install --upgrade pip \
    # Install all requirements
    && pip install -r requirements.txt \
    # Clean the path
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the entire project into the image
COPY . .

# Copy both the models just in case