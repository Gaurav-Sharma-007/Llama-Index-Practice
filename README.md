# LLamaIndex-Practice

A financial research assistant that combines LlamaIndex, AWS Bedrock, S3 Vectors, Indian stock data APIs, and a React chat frontend.

## What It Does

- Ingests `app/stock-research.pdf` into S3 Vectors with Bedrock embeddings.
- Retrieves relevant PDF chunks through the existing custom retriever in `app/retreiver.py`.
- Uses the existing LlamaIndex `FunctionAgent` for stock research, stock details, trending stocks, and market news.
- Adds a React chat UI with:
  - Initial assistant greeting: `Hi, How can I help you today ? `
  - In-chat memory so follow-up questions retain context.
  - A left vertical chat ribbon.
  - A `New Chat` button that starts a fresh conversation and clears memory for that chat.

## Project Structure

```text
.
├── api_server.py              # New FastAPI bridge for the React frontend
├── app/
│   ├── ingester.py            # PDF ingestion into S3 Vectors
│   ├── retreiver.py           # Existing backend agent and tools, unchanged
│   ├── test.py                # S3 vector listing helper
│   └── stock-research.pdf     # Source document
├── frontend/
│   ├── index.html
│   ├── package.json
│   └── src/
│       ├── main.jsx           # React chat app
│       └── styles.css         # Professional chat/search UI styling
├── main.py                    # Existing demo script
└── requirements.txt           # Python dependencies
```

## Backend Note

`app/retreiver.py` has not been changed. The new `api_server.py` loads that file as a bridge and skips the top-level sample `asyncio.run(main())` call so the frontend can use your existing `agent` through an HTTP endpoint.

## Prerequisites

- Python 3.11+
- Node.js 20+
- AWS credentials configured locally
- Access to the Bedrock and S3 Vectors resources used in `app/retreiver.py`

## Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Install Frontend Dependencies

```bash
cd frontend
npm install
```

## Ingest The PDF

Run ingestion whenever the research PDF changes:

```bash
cd app
python ingester.py
```

## Start The API Server

From the repository root:

```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

The chat endpoint is:

```text
POST http://localhost:8000/api/chat
```

## Start The React Frontend

In another terminal:

```bash
cd frontend
npm run dev
```

Open the local Vite URL, usually:

```text
http://localhost:5173
```

During development, Vite proxies frontend requests from `/api/*` to `http://localhost:8000`. This avoids browser networking issues when you open the UI through `localhost`, `127.0.0.1`, or a local network URL.

## Configure API URL

The frontend defaults to the same-origin API path:

```text
/api/chat
```

To bypass the Vite proxy and point it elsewhere:

```bash
cd frontend
VITE_API_URL=http://localhost:8000/api/chat npm run dev
```

## Chat Memory

Memory is kept in React state per chat session. When you send a message, the frontend sends the current chat history to `api_server.py`, and the API bridge wraps that history into the prompt before calling your existing LlamaIndex agent.

Clicking `New Chat` creates a clean conversation with no previous memory.

## Troubleshooting

- If the frontend cannot answer, make sure the API server is running on port `8000`.
- If AWS errors appear, confirm your credentials, Bedrock model access, S3 Vectors bucket, and index name.
- If `npm run dev` fails, run `npm install` inside `frontend`.
- If Python imports fail, install dependencies with `pip install -r requirements.txt`.
