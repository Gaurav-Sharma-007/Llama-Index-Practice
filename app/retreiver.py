import asyncio
import requests
import boto3
from typing import List,Dict,Any
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool


# =========================================================
# 1. BEDROCK LLM
# =========================================================
llm = BedrockConverse(
    model="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    #region_name="ap-south-1",
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

# Base URL for the stock API. This will be used to construct the full endpoint URLs for different API calls. The API provides various endpoints to fetch stock details, trending stocks, and market news, which can be accessed by appending the specific endpoint to this base URL.

BASE_URL = "https://stock.indianapi.in"

HEADERS = {
    "Accept": "application/json",
    "x-api-key": "sk-live-5kjUZn5CKD1f1vPoUtQUDKb1pVy74QVDwqFjoRtp"
}


def call_stock_api(endpoint: str, params: dict = None):
    url = f"{BASE_URL}/{endpoint}"

    response = requests.get(
        url,
        headers=HEADERS,
        params=params
    )

    response.raise_for_status()

    return response.json()

#Stock details API provides comprehensive information about a specific stock, including its current price, historical performance, financial ratios, and other relevant data. This can be useful for users who want to analyze a particular stock before making investment decisions. The tool can be invoked when users ask about specific stocks, their performance, or financial details.

def get_stock_details(symbol: str):
    return call_stock_api(
        "stock",
        params={"name": symbol}
    )


stock_tool = FunctionTool.from_defaults(
    fn=get_stock_details,
    name="get_stock_details",
    description="""
    Get detailed stock information.
    Input should be NSE/BSE stock symbol.
    """
)

def get_trending_stocks():
    return call_stock_api("trending")

#Trending stocks API provides a list of currently trending Indian stocks in the market. This can be useful for users who want to stay updated on market movers and hot stocks. The tool can be invoked when users ask about trending stocks, hot stocks, or market movers.

trending_tool = FunctionTool.from_defaults(
    fn=get_trending_stocks,
    name="get_trending_stocks",
    description="""
    Get trending Indian stocks in the market.
    Use when user asks:
    - trending stocks
    - hot stocks
    - market movers
    """
)

def get_market_news():
    return call_stock_api("news")


news_tool = FunctionTool.from_defaults(
    fn=get_market_news,
    name="get_market_news",
    description="Get latest Indian stock market news."
)


# =========================================================
# 8. CREATE AGENT
# =========================================================
agent = FunctionAgent(
    tools=[rag_tool, stock_tool,trending_tool,news_tool],
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
        "What does the report say about investment strategies, also I need to know what is the Tata Motors stock today ??"
    )
    print(response)

asyncio.run(main())