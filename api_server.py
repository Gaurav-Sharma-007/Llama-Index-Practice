import ast
import asyncio
import types
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

APP_DIR = Path(__file__).parent
RETRIEVER_PATH = APP_DIR / "app" / "retreiver.py"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str


def load_retriever_module() -> types.ModuleType:
    source = RETRIEVER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RETRIEVER_PATH))

    filtered_body = []
    for node in tree.body:
        is_auto_run = (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "run"
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "asyncio"
        )
        if not is_auto_run:
            filtered_body.append(node)

    tree.body = filtered_body
    ast.fix_missing_locations(tree)

    module = types.ModuleType("retreiver_api_bridge")
    module.__file__ = str(RETRIEVER_PATH)
    exec(compile(tree, str(RETRIEVER_PATH), "exec"), module.__dict__)
    return module


retriever_module = load_retriever_module()

app = FastAPI(title="LlamaIndex Financial Research API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|192\.168\.\d+\.\d+):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_context_prompt(message: str, history: list[ChatMessage]) -> str:
    recent_history = history[-16:]
    history_block = "\n".join(
        f"{item.role.capitalize()}: {item.content}" for item in recent_history
    )

    return (
        "Use this conversation history as context when it is relevant. "
        "If the previous messages are unrelated, answer the latest request directly.\n\n"
        f"Conversation history:\n{history_block}\n\n"
        f"Latest user request: {message}"
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    prompt = build_context_prompt(request.message, request.history)

    try:
        response = await retriever_module.agent.run(prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(answer=str(response))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
