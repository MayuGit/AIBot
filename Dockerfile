# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the dependency file and install libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Gradio application script
COPY app.py .

# Expose the Gradio default port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]