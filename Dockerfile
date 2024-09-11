# syntax=docker/dockerfile:1.2
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Copie requirements files
COPY requirements.txt .
COPY requirements-dev.txt .
COPY requirements-test.txt .

# Install dependencies
RUN apt-get update && \
    apt-get install -y build-essential gcc libssl-dev

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt
RUN pip install --no-cache-dir -r requirements-test.txt

# Copy the rest of the code into the container
COPY . .

# Expose the port the application will run on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
