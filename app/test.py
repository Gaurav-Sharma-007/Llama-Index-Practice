import boto3

# Create a S3 Vectors client in the AWS Region of your choice. 
s3vectors = boto3.client("s3vectors", region_name="ap-south-1")

#List vector indexes in your vector bucket
response = s3vectors.list_indexes(vectorBucketName="llamaindex-bucket")
indexes = response["indexes"]
print(indexes)