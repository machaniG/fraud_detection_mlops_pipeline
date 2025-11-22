# Use a slim Python image for smaller size
FROM python:3.10-slim

# Set environment variables for non-interactive commands
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set MLflow tracking URI for the container environment
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5001 
# Note: For local testing, this will use http://localhost:5001.
# In a real setup, this would be an actual domain/IP of the MLflow server.

# Set PYTHONPATH to include the /app directory so 'src' imports work
ENV PYTHONPATH=/app

# Install dependencies
WORKDIR /app
COPY requirements.txt /app/

# Install Python dependencies, optimizing for cached layers
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# The dataset needs to be here for the script to run
# NOTE: In a real-world scenario, the data is typically pulled from S3 or a DB,
# but for this demo, we assume the data file is copied into the container.
COPY transactions.csv /app/transactions.csv 

# Command to run the training script when the container starts
# The data file path is passed as an argument.
ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--data", "transactions.csv"]