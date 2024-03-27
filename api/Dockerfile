# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy all contents into the container at /app
COPY . /app

# Create and activate a virtual environment
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Copy the trained model file into the container
COPY C:\Users\afshi\Documents\GitHub\immo-eliza-deployment\api\Updated_trained_catboost_model.pkl /app


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run your application using uvicorn
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "--reload", "app:app"]
