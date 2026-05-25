import boto3
import json

from llama_index.embeddings.bedrock import BedrockEmbedding


# =====================================================
# 1. CREATE EMBEDDING MODEL
# =====================================================

embed_model = BedrockEmbedding(
    model_name="amazon.titan-embed-text-v2:0",
    region_name="ap-south-1",
)


# =====================================================
# 2. USER QUERY
# =====================================================

query = "What does the report say about Apple stock growth?"


# =====================================================
# 3. CONVERT QUERY INTO EMBEDDING VECTOR
# =====================================================

query_embedding = embed_model.get_text_embedding(query)

print(f"Embedding dimensions: {len(query_embedding)}")


# =====================================================
# 4. CONNECT TO S3 VECTORS
# =====================================================

session = boto3.Session(profile_name="default")

s3vectors = session.client(
    "s3vectors",
    region_name="ap-south-1"
)


# =====================================================
# 5. SEARCH SIMILAR VECTORS
# =====================================================

response = s3vectors.query_vectors(
    vectorBucketName="llamaindex-bucket",
    indexName="indexer",
    queryVector={
        "float32": query_embedding
    },
    topK=5
)


# =====================================================
# 6. PRINT RESULTS
# =====================================================

print(json.dumps(response, indent=2))