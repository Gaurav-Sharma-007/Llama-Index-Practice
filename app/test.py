import boto3
import json

session = boto3.Session(profile_name="default")

s3vectors = session.client(
    "s3vectors",
    region_name="ap-south-1"
)

response = s3vectors.list_vectors(
    vectorBucketName="llamaindex-bucket",
    indexName="indexer"
)

print(json.dumps(response, indent=2))