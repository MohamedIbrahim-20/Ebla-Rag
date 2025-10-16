# Ebla-RAG

**Ebla-RAG** is a modular Retrieval-Augmented Generation (RAG) system combining LangChain and LlamaIndex for context-aware chat and document Q&A. It features session-based chat, semantic search, and project-based document management, powered by FastAPI, Groq LLM, FAISS, and SQLite.

---

## 🚀 Features
- **Dual RAG Engines:** LangChain & LlamaIndex integration
- **Session-Based Chat:** Persistent, context-rich conversations
- **Semantic Search:** Vector-based document retrieval
- **Project Isolation:** Multi-project support
- **Document Upload & Indexing:** CSV and file support
- **Summarization:** Automatic context compression for long chats
- **Modern Web UI:** Responsive chat interface
- **API-first Design:** RESTful endpoints for all operations

---

## 🏗️ System Architecture
See the full architecture diagram: [system_architecture.svg](system_architecture.svg)

- **Frontend:** HTML/Jinja2, JavaScript, CSS
- **API Gateway:** FastAPI routers
- **Business Logic:** Controllers for RAG, data, projects, and processing
- **Data Layer:** SQLite, FAISS, LlamaIndex
- **External Services:** Groq LLM, HuggingFace embeddings

For detailed documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## ⚡ Quickstart

### Local Development
1. **Clone the repo:**
   ```sh
   git clone https://github.com/your-org/Ebla-RAG.git
   cd Ebla-RAG
   ```
2. **Install dependencies:**
   ```sh
   pip install -r src/requirements.txt
   ```
3. **Configure environment:**
   - Copy `.env.example` to `.env` and set your API keys and config.
4. **Run the app:**
   ```sh
   uvicorn src.main:app --reload
   ```
5. **Open in browser:**
   - Visit [http://localhost:8000/](http://localhost:8000/)


---

## 🛠️ API Endpoints
See [API_ENDPOINTS.md](src/API_ENDPOINTS.md) for full details.

- `POST /api/v1/chat` — Chat with RAG context
- `POST /api/v1/data/index/{project_id}` — Index documents
- `POST /api/v1/data/search/{project_id}` — Search documents
- `POST /api/v1/data/ask/{project_id}` — Ask questions
- `GET /api/v1/history/{session_id}` — Get chat history
- `GET /api/v1/data/status/{project_id}` — System status

---

## 💡 Usage Examples

### Chat with Context
```python
import requests
resp = requests.post('http://localhost:8000/api/v1/chat', json={
    "session_id": "your-session-id",
    "message": "What is quantum computing?",
    "project_id": "science"
})
print(resp.json())
```

### Index a CSV Document
```python
resp = requests.post('http://localhost:8000/api/v1/data/index/science', json={
    "csv_file": "science_dataset.csv"
})
print(resp.json())
```

---

## 📁 Directory Structure
```
Ebla-RAG/
├── src/
│   ├── controllers/      # Business logic (RAG, Data, Project, Process)
│   ├── helpers/          # Config and prompt management
│   ├── models/           # Database and ORM
│   ├── routes/           # FastAPI routers
│   ├── static/           # Frontend assets (css, js)
│   ├── templates/        # Jinja2 HTML templates
│   ├── main.py           # App entry point
│   ├── requirements.txt  # Python dependencies
│   └── API_ENDPOINTS.md  # API documentation
├── ARCHITECTURE.md       # System architecture docs
├── system_architecture.svg# Architecture diagram
├── README.md             # This file
└── test_data/            # Example datasets
```
