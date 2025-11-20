# Cisco Agent Configuration Generator

A Retrieval-Augmented Generation (RAG) pipeline that ingests technical SQL + text specifications and generates configuration files or code snippets based on user instructions.

## 🏗️ Architecture

- **Backend**: Python 3.11 + Flask
- **Database**: MongoDB Atlas Cloud (vector search enabled)
- **Vector Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **LLM Generation**: Google Gemini (gemini-2.5-flash)
- **Frontend**: React single-page app

## 📋 Prerequisites

- Python 3.11+
- Node.js 16+ and npm
- MongoDB Atlas account with vector search enabled

## 🚀 Setup Instructions

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory (already created with default values):

```env
MONGODB_URI="mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority"
MONGODB_DB="ragdb"
MONGODB_COLLECTION="chunks"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
GEMINI_API_KEY="your_gemini_api_key_here"
GEMINI_MODEL="gemini-2.5-flash"
GEN_TEMPERATURE="0.2"
TOP_K="5"
```

**Note**: Update `MONGODB_URI` with your actual MongoDB Atlas cluster hostname.

### 3. MongoDB Vector Index Setup

In MongoDB Atlas, create a vector search index on your collection:

1. Go to Atlas Search → Create Search Index
2. Use JSON Editor and paste:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

3. Name the index: `idx_embedding`

### 4. Frontend Setup

```bash
cd frontend
npm install
```

### 5. Run the Application

**Terminal 1 - Backend:**
```bash
cd backend
python app.py
```

**Terminal 2 - Frontend (development):**
```bash
cd frontend
npm start
```

Or build for production:
```bash
cd frontend
npm run build
```

The backend will serve the frontend from the build folder when running in production mode.

## 📡 API Endpoints

### `GET /api/health`
Health check endpoint to verify MongoDB connection.

**Response:**
```json
{
  "status": "healthy",
  "mongodb": "connected",
  "database": "ragdb",
  "collection": "chunks"
}
```

### `POST /api/ingest`
Upload and ingest SQL or text files.

**Request:** Multipart form data with `file` field (.sql or .txt)

**Response:**
```json
{
  "status": "success",
  "message": "Ingested 5 chunks from example.sql",
  "chunks": 5,
  "source_id": "example.sql"
}
```

### `POST /api/generate-config`
Generate configuration based on user instruction.

**Request:**
```json
{
  "instruction": "Generate YAML config for firewall rule X"
}
```

**Response:**
```json
{
  "config_output": "generated config code...",
  "references": [
    {
      "chunk_id": "...",
      "source_id": "example.sql",
      "text_preview": "...",
      "score": 0.95
    }
  ],
  "full_response": "..."
}
```

## 🎯 Usage

1. **Upload Specifications**: Use the upload section to upload `.sql` or `.txt` files containing technical specifications, commands, or expected configurations.

2. **Generate Configurations**: Enter an instruction in the textarea (e.g., "Generate YAML for command X") and click "Generate Config". The system will:
   - Embed your instruction
   - Retrieve the most relevant chunks from uploaded specs
   - Generate a configuration using the LLM
   - Display the result with context references

## 🔧 Features

- **Intelligent Chunking**: Automatically chunks SQL files by statements or text files by ~1000 characters
- **Vector Search**: Uses MongoDB Atlas vector search for semantic retrieval
- **Context-Aware Generation**: LLM generates configurations based on retrieved context
- **Reference Tracking**: Shows which chunks were used for generation
- **Fallback Mechanism**: Manual cosine similarity if vector index is not available

## ⚠️ Notes

- The first run will download the embedding model (~90MB)
- Ensure you have a valid GEMINI_API_KEY
- MongoDB Atlas vector search requires a dedicated cluster with vector search enabled

## 🐛 Troubleshooting

- **MongoDB Connection Issues**: Verify your connection string and network access in Atlas
- **Vector Search Not Working**: Check that the vector index `idx_embedding` is created
- **Model Loading Errors**: Ensure sufficient disk space and RAM
- **CORS Errors**: Make sure Flask-CORS is installed and configured

## 📝 License

MIT

