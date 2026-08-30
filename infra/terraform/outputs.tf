output "s3_bucket_name" {
  description = "Name of the created S3 bucket."
  value       = aws_s3_bucket.artifacts.id
}

output "s3_bucket_arn" {
  description = "ARN of the created S3 bucket."
  value       = aws_s3_bucket.artifacts.arn
}

output "ecr_repository_url" {
  description = "URI of the created ECR repository."
  value       = aws_ecr_repository.pipeline.repository_url
}

output "ecr_repository_arn" {
  description = "ARN of the created ECR repository."
  value       = aws_ecr_repository.pipeline.arn
}