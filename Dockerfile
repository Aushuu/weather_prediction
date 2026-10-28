FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose default Streamlit port
EXPOSE 8501 10000

ENV PORT=8501

# Launch Streamlit with dynamic port binding for Render / Railway / Cloud
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
