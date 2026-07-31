FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# libgomp1 is required by XGBoost at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer is cached across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the project and install the package itself. Editable (-e) keeps the
# package rooted at /app so config's data/ and models/ paths resolve correctly.
COPY . .
RUN pip install --no-cache-dir -e . --no-deps

# Run as a non-root user.
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 8000 = FastAPI, 8501 = Streamlit.
EXPOSE 8000 8501

# Default: serve the API. The dashboard service overrides this command.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
