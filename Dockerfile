# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install numpy pandas scikit-learn catboost fastapi uvicorn joblib

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run your application using uvicorn
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "--reload", "api.app:app"]
