# mlops-end-to-end-pipeline-automation

A production-grade, modular MLOps pipeline demonstrating:

- Clean Python software design (type hints, docstrings, structured logging, custom exceptions, decoupled config)
- **MLflow** experiment tracking and Model Registry automation
- **AWS S3** integration for raw/processed artifact storage (`boto3`)
- Automated **quality gates** that block bad models from being registered
- A single **CLI orchestrator** (`main.py`) chaining every stage

## Directory Structure

```
mlops-end-to-end-pipeline-automation/
├── config/
│   └── config.yaml            # All tunable settings: paths, S3, hyperparams, quality gates
├── src/
│   ├── config.py               # Decoupled YAML config loader (dot-access)
│   ├── logger.py                # Centralized structured logging
│   ├── exceptions.py            # Custom exception hierarchy
│   ├── data_ingestion.py        # Load, clean, split, S3 upload/download
│   ├── train.py                 # Model training + MLflow run/logging
│   ├── evaluate.py               # Metric computation + quality-gate checks
│   └── register.py               # MlflowClient-based registry automation
├── pipelines/                   # Reserved for higher-level DAG orchestration (Airflow/Prefect)
├── tests/                       # Pytest unit tests
├── scripts/
│   └── generate_sample_data.py  # Creates a synthetic dataset for a first run
├── main.py                       # Click-based CLI entrypoint
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment (AWS creds, MLflow URI)
cp .env.example .env
# edit .env with real values, or leave S3 disabled in config.yaml for local-only runs

# 3. Generate a synthetic dataset (skip if you have your own raw_dataset.csv)
python scripts/generate_sample_data.py

# 4. Run the full pipeline: ingest -> train -> evaluate -> register
python main.py --run-all
```

## Running Individual Stages

```bash
python main.py --ingest              # data ingestion only
python main.py --train               # ingestion + training
python main.py --evaluate            # ingestion + training + evaluation
python main.py --register            # full pipeline incl. registration (same as --run-all)
python main.py --config path/to/other_config.yaml --run-all
```

## Quality Gates

Defined in `config/config.yaml` under `quality_gates`. A trained model is registered
to the MLflow Model Registry (stage: `Staging` by default) **only if** it meets or
exceeds every threshold:

```yaml
quality_gates:
  min_accuracy: 0.80
  min_f1_score: 0.75
  min_precision: 0.70
  min_recall: 0.70
```

If any threshold is not met, `evaluate.py` returns `passed_gates: False`, `register.py`
skips registration entirely, and the pipeline logs the specific failed checks.

## Viewing Experiments

```bash
mlflow ui --backend-store-uri mlruns
```

Then open `http://localhost:5000` to browse runs, metrics, and registered model versions.

## Running Tests

```bash
pytest tests/ -v --cov=src
```

## Disabling S3 (local-only mode)

Set `s3.enabled: false` in `config/config.yaml` — `data_ingestion.py` will skip all
upload/download calls and log a warning instead of failing.

## Continuous Integration

`.github/workflows/ci.yml` runs on every push/PR to `main`:

1. Lints with `ruff` and runs the `pytest` suite with coverage, across Python 3.10 and 3.11.
2. Runs a full pipeline smoke test on synthetic data (S3 disabled) and uploads the
   resulting `mlruns/` directory as a build artifact.

## Docker

Build and run the pipeline in a container:

```bash
docker build -t mlops-pipeline .
docker run --rm \
  -v "$(pwd)/mlruns:/app/mlruns" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  --env-file .env \
  mlops-pipeline --run-all
```

Mounting `mlruns/`, `data/`, and `artifacts/` keeps experiment history and model
artifacts on the host instead of trapped inside the container.

## Local MLflow + Postgres Stack (docker-compose)

`docker-compose.yml` spins up a real tracking server backed by Postgres instead of
the local `mlruns/` folder, so runs and registered models persist and are browsable
from a proper MLflow UI:

```bash
docker compose up -d mlflow-db mlflow-server   # start Postgres + MLflow tracking server
open http://localhost:5000                      # browse the MLflow UI

docker compose run --rm pipeline --run-all       # run the full pipeline against it
docker compose run --rm pipeline --ingest        # or an individual stage
```

`mlflow-server` uses `docker/mlflow.Dockerfile` (mlflow + psycopg2 + boto3) and stores
artifacts in a named volume (`mlflow-artifacts`); swap `--default-artifact-root` in
`docker-compose.yml` for an `s3://...` URI to use S3 as the artifact store instead.

## Deployment Notes

This repo is a training/registration pipeline, not a live inference service, so
"deployment" here means running it reliably and automatically rather than standing up
a web server. What's included:

- **`.github/workflows/scheduled-retrain.yml`**: runs the full pipeline on a cron
  schedule (default: weekly) against the real S3 bucket / MLflow server via GitHub
  Secrets, letting the quality gates decide whether to register each new model.
- **`.github/workflows/deploy-ecr.yml`**: after CI passes on `main`, builds the
  Docker image and pushes it to Amazon ECR, tagged with both `latest` and the commit
  SHA — the artifact a scheduler (ECS Fargate task, Kubernetes CronJob, etc.) would run.
- **`docker-compose.yml`**: a real MLflow tracking server + Postgres backend for local
  development, so runs aren't limited to a local `mlruns/` folder (see above).
- **Serving the registered model**: once a model is in the `Staging`/`Production`
  stage of the registry, serve it separately with `mlflow models serve -m
  "models:/<registered_model_name>/Staging"` or deploy it to SageMaker/a container —
  this pipeline's job ends at registration.

### Required GitHub Secrets

For `scheduled-retrain.yml` and `deploy-ecr.yml` to run, add these under
**Settings → Secrets and variables → Actions** (a `production` environment with its
own secrets is recommended over repo-level secrets):

| Secret | Used by | Purpose |
|---|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | both | AWS auth for S3 + ECR |
| `AWS_REGION` | both | AWS region for S3 + ECR |
| `S3_BUCKET_NAME` | scheduled-retrain | bucket holding `raw_dataset.csv` |
| `MLFLOW_TRACKING_URI` | scheduled-retrain | real tracking server URL |
