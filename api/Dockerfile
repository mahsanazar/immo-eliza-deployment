# Use an official Python runtime as a parent image
FROM python:3.11-slim

RUN mkdir /app


# Set the working directory to /app
WORKDIR /app

COPY . /app

# Update pip
RUN pip install --upgrade pip

# Install dependencies from "requirements.txt"
RUN pip install -r requirements.txt

# Create and activate a virtual environment
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Command to run your application using uvicorn
CMD uvicorn --host 0.0.0.0 --port 8000 app:app

