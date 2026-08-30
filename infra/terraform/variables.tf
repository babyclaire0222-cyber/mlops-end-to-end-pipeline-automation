variable "aws_region" {
  description = "AWS region to create resources in."
  type        = string
  default     = "us-east-1"
}

variable "s3_bucket_name" {
  description = "Globally-unique name for the S3 bucket holding pipeline artifacts. Must match config/config.yaml s3.bucket_name."
  type        = string
}

variable "ecr_repository_name" {
  description = "Name for the ECR repository holding the pipeline Docker image. Must match ECR_REPOSITORY in .github/workflows/deploy-ecr.yml."
  type        = string
  default     = "mlops-end-to-end-pipeline-automation"
}

variable "environment" {
  description = "Environment tag applied to created resources (e.g. dev, staging, production)."
  type        = string
  default     = "production"
}