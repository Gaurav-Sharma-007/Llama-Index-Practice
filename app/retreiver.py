import asyncio
import boto3
from typing import List
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.agent.workflow import FunctionAgent

# =========================================================
# 1. BEDROCK LLM
# =========================================================
llm = BedrockConverse(
    model="deepseek.v3.2",
    region_name="ap-south-1",
)

# =========================================================
# 2. EMBEDDING MODEL
# =========================================================
embed_model = BedrockEmbedding(
    model_name="amazon.titan-embed-text-v2:0",
    region_name="ap-south-1",
)

# =========================================================
# 3. S3 VECTOR CLIENT
# =========================================================
session = boto3.Session(profile_name="default")
s3vectors = session.client(
    "s3vectors",
    region_name="ap-south-1"
)

# =========================================================
# 4. CUSTOM RETRIEVER
# =========================================================
class S3VectorRetriever(BaseRetriever):
    def _retrieve(self, query_bundle):
        # ---------------------------------------------
        # Convert query into embedding
        # ---------------------------------------------
        query_embedding = embed_model.get_text_embedding(
            query_bundle.query_str
        )

        # ---------------------------------------------
        # Query S3 Vectors
        # ---------------------------------------------
        response = s3vectors.query_vectors(
            vectorBucketName="llamaindex-bucket",
            indexName="indexer",
            queryVector={
                "float32": query_embedding
            },
            topK=5,
            returnMetadata=True
        )

        retrieved_nodes = []

        # ---------------------------------------------
        # Convert AWS response into LlamaIndex nodes
        # ---------------------------------------------
        for match in response["vectors"]:
            text = match["metadata"]["text"]
            node = TextNode(text=text)
            scored_node = NodeWithScore(
                node=node,
                score=1 - match.get("distance", 0)
            )
            retrieved_nodes.append(scored_node)

        return retrieved_nodes

# =========================================================
# 5. CREATE RETRIEVER
# =========================================================
retriever = S3VectorRetriever()

# =========================================================
# 6. CREATE QUERY ENGINE
# =========================================================
response_synthesizer = get_response_synthesizer(llm=llm)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# =========================================================
# 7. CONVERT TO TOOL
# =========================================================
rag_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="stock_research_tool",
    description=(
        "Useful for answering questions "
        "about stock research PDFs and market analysis."
    )
)

# =========================================================
# 8. CREATE AGENT
# =========================================================
agent = FunctionAgent(
    tools=[rag_tool],
    llm=llm,
    system_prompt=(
        "You are a financial research assistant."
    )
)

# =========================================================
# 9. ASK QUESTIONS
# =========================================================
async def main():
    response = await agent.run(
        "What does the report say about investment strategies?"
    )
    print(response)

asyncio.run(main())