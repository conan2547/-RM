FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement) — ต้องสร้างก่อน
RUN useradd -m -u 1000 user

# Copy requirements and install Python dependencies
COPY web/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy web application
COPY web/ ./

# สร้าง cache directories ให้ user มีสิทธิ์เขียน
RUN mkdir -p /home/user/.cache/huggingface && \
    chown -R user:user /home/user/.cache && \
    chown -R user:user /app

USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    HF_HOME=/home/user/.cache/huggingface

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Run the app on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
