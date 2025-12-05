# Use a Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This includes app.py and the data/ folder
COPY . .

# Set the port that the container will expose (Cloud Run default)
EXPOSE 8080

# Command to run the Streamlit app when the container starts
# The crucial --server.port 8080 and --server.address 0.0.0.0 flags are necessary for Cloud Run.
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]