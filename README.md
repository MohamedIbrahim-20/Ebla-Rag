# Ebla-RAG

A context-aware chat application using Retrieval-Augmented Generation (RAG) to provide accurate, document-grounded responses.

## Features

- Document indexing with vector embeddings (FAISS)
- Semantic search for relevant context
- Chat history with session management
- Context-aware conversations with summarization
- Multiple RAG implementations (LangChain and LlamaIndex)
- FastAPI backend with documented endpoints

## Quick Start

### Requirements

- Python 3.8+
- Groq API key

### Setup

1. Clone the repository and navigate to the project directory

2. Create a Python environment:
   ```bash
   conda create -n ebla-rag python=3.8
   conda activate ebla-rag
   ```

3. Install dependencies:
   ```bash
   pip install -r src/requirements.txt
   ```

4. Configure environment:
   ```bash
   cp src/.env.example src/.env
   ```
   Edit the `.env` file to add your Groq API key

5. Start the server:
   ```bash
   cd src
   uvicorn main:app --reload --port 5000
   ```

## API Usage

The API provides endpoints for:

- Indexing documents: `POST /api/v1/data/index/{project_id}`
- Searching documents: `POST /api/v1/data/search/{project_id}`
- Asking questions: `POST /api/v1/data/ask/{project_id}`
- Chat with context: `POST /api/v1/chat`
- View chat history: `GET /api/v1/history/{session_id}`

See `API_ENDPOINTS.md` for detailed documentation.

## Testing

Run the test script to verify your setup:
```bash
python test_simple.py
```

## Architecture

Ebla-RAG implements a modular architecture with:
- SQLite database for chat history
- FAISS vector store for document embeddings
- Groq LLM integration for text generation
- FastAPI for RESTful endpoints