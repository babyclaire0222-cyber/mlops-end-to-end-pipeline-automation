# Minimal MLflow tracking server image with Postgres + S3 artifact support.
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==2.14.1 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.144

EXPOSE 5000

# backend-store-uri and default-artifact-root are supplied at runtime via
# command args in docker-compose.yml so they can point at different
# databases/buckets per environment without rebuilding the image.
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
