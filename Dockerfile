# ABD Spirits Supply Planner - API + dashboard
FROM python:3.11-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY data/processed/forecast_2026_11.parquet data/processed/store_dim.parquet data/processed/county_profile.parquet data/processed/
COPY reports/tables/monitoring.json reports/tables/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
