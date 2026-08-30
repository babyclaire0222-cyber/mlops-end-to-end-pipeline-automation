# Infrastructure as code for mlops-end-to-end-pipeline-automation.
#
# Codifies the two AWS resources this project depends on:
#   - an S3 bucket for raw/processed data + model artifacts
#   - an ECR repository for the pipeline's Docker image
#
# These were originally created manually via the AWS CLI/console for this
# project; this configuration lets you recreate them reproducibly (e.g. in
# a fresh AWS account) or manage them going forward via `terraform plan`/
# `terraform apply` instead of manual console clicks.

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ------------------------------------------------------------------------------
# S3 bucket - raw data, processed data, and model artifacts
# ------------------------------------------------------------------------------
resource "aws_s3_bucket" "artifacts" {
  bucket = var.s3_bucket_name

  tags = {
    Project     = "mlops-end-to-end-pipeline-automation"
    ManagedBy   = "terraform"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ------------------------------------------------------------------------------
# ECR repository - Docker image for the pipeline
# ------------------------------------------------------------------------------
resource "aws_ecr_repository" "pipeline" {
  name                 = var.ecr_repository_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Project     = "mlops-end-to-end-pipeline-automation"
    ManagedBy   = "terraform"
    Environment = var.environment
  }
}

# Automatically expire untagged images after N days to control storage cost.
resource "aws_ecr_lifecycle_policy" "pipeline" {
  repository = aws_ecr_repository.pipeline.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images after 14 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 14
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}