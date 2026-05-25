import boto3
import json
import uuid
import os

from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter


# =========================================================
# 1. LOAD PDF
# =========================================================

loader = PDFReader()
docs = loader.load_data(file="./stock-research.pdf")

print(f"Loaded {len(docs)} pages")


# =========================================================
# 2. SPLIT DOCUMENT INTO CHUNKS
# =========================================================

parser = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

nodes = parser.get_nodes_from_documents(docs)

print(f"Generated {len(nodes)} chunks")


# =========================================================
# 3. CREATE BEDROCK EMBEDDING MODEL
# =========================================================

embed_model = BedrockEmbedding(
    model_name="amazon.titan-embed-text-v2:0",
    region_name="ap-south-1",
)


# =========================================================
# 4. GENERATE EMBEDDINGS
# =========================================================

texts = [
    node.get_content(metadata_mode="all")
    for node in nodes
]

embeddings = embed_model.get_text_embedding_batch(
    texts,
    show_progress=True
)

print(f"Generated {len(embeddings)} embeddings")


# =========================================================
# 5. ATTACH EMBEDDINGS BACK TO NODES
# =========================================================

for node, embedding in zip(nodes, embeddings):
    node.embedding = embedding


# =========================================================
# 6. CREATE S3 VECTORS CLIENT
# =========================================================

session = boto3.Session(profile_name="default")

s3vectors = session.client(
    "s3vectors",
    region_name="ap-south-1"
)


# =========================================================
# 7. CONFIGURATION
# =========================================================

VECTOR_BUCKET_NAME = os.getenv("VECTOR_BUCKET_NAME", "llamaindex-bucket")
INDEX_NAME = os.getenv("INDEX_NAME", "indexer")


# =========================================================
# 8. UPSERT VECTORS INTO S3 VECTOR INDEX
# =========================================================

vectors = []

for i, node in enumerate(nodes):

    vector_item = {
        "key": str(uuid.uuid4()),

        # actual embedding vector
        "data": {
            "float32": node.embedding
        },

        # metadata
        "metadata": {
            "text": node.text[:1000],
            "chunk_id": str(i),
            "source": "stock-research.pdf"
        }
    }

    vectors.append(vector_item)


# Batch upload
response = s3vectors.put_vectors(
    vectorBucketName=VECTOR_BUCKET_NAME,
    indexName=INDEX_NAME,
    vectors=vectors
)

print("Vectors uploaded successfully")
print(json.dumps(response, indent=2))