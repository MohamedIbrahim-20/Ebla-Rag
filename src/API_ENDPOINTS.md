# RAG API Endpoints

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   python setup_env.py
   # Edit .env file and add your GROQ_API_KEY
   # The system uses pydantic-settings to load configuration
   ```

3. **Start the server:**
   ```bash
   uvicorn main:app --reload --port 5000
   ```

## API Endpoints

### 1. System Status
**GET** `/api/v1/data/status/{project_id}`

Check system status and configuration.

**Response:**
```json
{
  "project_id": "2",
  "groq_api_configured": true,
  "models_initialized": false,
  "langchain_index_exists": false,
  "llamaindex_exists": false,
  "llm_model": "llama-3.3-70b-versatile",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

### 2. Index Documents
**POST** `/api/v1/data/index/{project_id}`

Index documents from CSV file for RAG.

**Request Body:**
```json
{
  "csv_file": "tech_100_long_real.csv"
}
```

**Response:**
```json
{
  "message": "Documents indexed successfully",
  "project_id": "2",
  "csv_file": "tech_100_long_real.csv"
}
```

### 3. Search Documents
**POST** `/api/v1/data/search/{project_id}`

Search for relevant documents.

**Request Body:**
```json
{
  "query": "transistor invention",
  "method": "langchain",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "transistor invention",
  "method": "langchain",
  "results": [
    {
      "content": "Point-contact transistor demonstrated at Bell Labs...",
      "metadata": {...},
      "score": 0.85
    }
  ],
  "total_results": 3
}
```

### 4. Ask Question (RAG)
**POST** `/api/v1/data/ask/{project_id}`

Ask questions using RAG (Retrieval-Augmented Generation).

**Request Body:**
```json
{
  "question": "Who invented the transistor and when?",
  "method": "langchain"
}
```

**Response:**
```json
{
  "answer": "The transistor was invented by John Bardeen, Walter Brattain, and William Shockley at Bell Labs on December 23, 1947...",
  "retrieved_documents": [
    {
      "content": "Point-contact transistor demonstrated at Bell Labs...",
      "metadata": {...}
    }
  ],
  "method": "langchain"
}
```

### 5. Chat (Context-Aware with History)
**POST** `/api/v1/chat`

Create or continue a session; uses history + (optional) rolling summary + RAG to answer.

**Request Body (new session):**
```json
{
  "message": "Who invented the transistor and when?",
  "method": "langchain"
}
```

**Request Body (continue existing session):**
```json
{
  "session_id": "<paste-session-id>",
  "message": "Summarize it in one sentence.",
  "method": "langchain"
}
```

**Response:**
```json
{
  "session_id": "...",
  "answer": "...",
  "retrieved_documents": [ { "content": "...", "metadata": {"id": "..."}, "score": 0.87 } ]
}
```

Notes:
- Retrieval method can be "langchain" or "llamaindex".
- A rolling summary is created automatically when sessions grow; summary is reused in future prompts.

### 6. Get Chat History
**GET** `/api/v1/history/{session_id}`

Returns session messages (ordered) and latest summary (if any).

**Response:**
```json
{
  "session_id": "...",
  "summary": "Compact summary of earlier turns or null",
  "messages": [
    {"id": "...", "role": "user", "content": "...", "created_at": "..."},
    {"id": "...", "role": "assistant", "content": "...", "created_at": "..."}
  ]
}
```

## Postman Collection

An importable collection is available at `src/postman_collection.json`.

Recommended order:
1. Status
2. Index CSV (run once)
3. Chat (new session)
4. Chat (continue session)
5. History

### 5. File Upload
**POST** `/api/v1/data/upload/{project_id}`

Upload files for processing.

**Form Data:**
- `file`: The file to upload

**Response:**
```json
{
  "message": "File uploaded successfully",
  "file_id": "abc123_test.txt"
}
```

### 6. Process Files
**POST** `/api/v1/data/process/{project_id}`

Process uploaded files into chunks.

**Request Body:**
```json
{
  "file_id": "abc123_test.txt",
  "chunk_size": 100,
  "overlap_size": 20
}
```

## Testing

Run the simple test script:
```bash
python test_simple.py
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `500`: Internal Server Error (check logs)

Common errors:
- `GROQ_API_KEY not found`: Set your Groq API key in .env file
- `LangChain not available`: Install required packages
- `File not found`: Ensure CSV file exists in project directory
