# Python version
FROM python:3.11-slim

EXPOSE 8501

# Set working directory
WORKDIR /mlops-td

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest o f the application
COPY . .

# Command to run the application
#CMD ["uvicorn", "--reload", "--host", "0.0.0.0", "app:app"]
#CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
CMD ["streamlit", "run", "app.py", "8501:8501", "--server.address=0.0.0.0"]