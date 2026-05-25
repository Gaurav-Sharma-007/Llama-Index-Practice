import boto3,os,json,uuid

from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter


loader = PDFReader(file_path="./stock-research.pdf")
docs = loader.load_data()
parser = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
nodes = parser.get_nodes_from_documents(docs)
print(nodes[0].text)

embed_model = BedrockEmbedding(
    model_name="amazon.titan-embed-text-v2:0",
    region_name="ap-south-1",
)

# Optional: configure your AWS credentials
session = boto3.Session(profile_name="default")

texts = [node.get_content(metadata_mode="all") for node in nodes]
embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)

# Attach each vector back onto its node
for node, embedding in zip(nodes, embeddings):
    node.embedding = embedding