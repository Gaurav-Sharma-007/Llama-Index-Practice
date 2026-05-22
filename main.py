import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.azure_openai import AzureOpenAI
import os

subscription_key = os.getenv("AZURE_OPENAI_KEY")

llms = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://gaurav-testing.openai.azure.com",
    api_key=subscription_key,
    engine="gpt-5.2",
)


# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    tools=[multiply],
    llm=llms,
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)


async def main():
    # Run the agent
    response = await agent.run("What is 1234 * 4567?")
    print(str(response))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())