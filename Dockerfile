# Use official Python slim image
FROM python:3.12-slim

# Set workdir
WORKDIR /app

# Copy only necessary files for faster builds
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0"]
