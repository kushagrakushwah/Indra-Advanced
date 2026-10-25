FROM python:3.12-slim

# Install system libraries required by OpenCV, rapidocr, and pypdfium2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user with UID 1000 (standard for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install CPU PyTorch and Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy repository code and install package in editable mode
COPY --chown=user:user . .
RUN pip install --no-cache-dir -e .

# Expose Hugging Face Space port
EXPOSE 7860

# Start server
CMD ["uvicorn", "indra_advanced.server:app", "--host", "0.0.0.0", "--port", "7860"]
