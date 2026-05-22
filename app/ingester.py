import boto3

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
bedrock_client = session.client("bedrock-runtime", region_name="ap-south-1")