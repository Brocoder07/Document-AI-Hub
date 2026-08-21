# Document AI Hub

Document AI Hub is a production-grade, highly sophisticated Retrieval-Augmented Generation (RAG) application specifically designed to process, index, and query massive telecommunications standards documentation (e.g., 3GPP TS and TR specifications) with a focus on **minimal to near-zero hallucinations**. 

The system provides role-based access, supporting various specialized domains (Telecom, Legal, Healthcare, Finance, Academic, Business) via a Strategy Pattern, with telecom being the primary focus. It utilizes hybrid search, cross-encoder re-ranking, and post-generation hallucination validation to ensure absolute accuracy in technical domains.

---

## 🏗 System Architecture

The application is built on a modern, highly decoupled architecture designed for scalability and accuracy.

### 1. Frontend (User Interface)
- **Framework**: Built with **Streamlit** for rapid, reactive UI development.
- **Key Modules**:
  - `main.py`: Entry point, manages routing and session restoration.
  - `session_state.py`: Manages global state (tokens, chat history, views).
  - `api_client.py`: Robust HTTP client for interacting with the backend APIs.
  - `components/`: Modular UI components (`auth.py`, `chat.py`, `documents.py`, `sidebar.py`).
- **Features**: User authentication, document library (upload/delete), role-aware chat interface displaying retrieved context, processing metrics, confidence scores, and hallucination risks.

### 2. Backend (API Server)
- **Framework**: Built with **FastAPI**. Exposes asynchronous RESTful endpoints.
- **Structure**:
  - `app/api/`: Route definitions (`auth.py`, `upload.py`, `search.py`, `rag.py`, etc.).
  - `app/services/`: Core business logic decoupled from API routing.
  - `app/core/`: Application config, security (JWT, Fernet encryption), and the custom LLM wrapper.
- **Features**: JWT-based authentication, AES message encryption (Fernet), background task processing for heavy document ingestion, and rate-limiting.

### 3. Databases
- **Relational Database**: **PostgreSQL** managed via **SQLAlchemy** (ORM) and **Alembic** (migrations).
  - *Models*: `User` (roles, auth), `Document` (tracking uploads, status, deduplication via hashing), `ChatSession`, `ChatMessage` (encrypted history).
- **Vector Database**: **Weaviate** (Dockerized).
  - Stores document chunks and embeddings.
  - Supports rich metadata filtering (user_id, file_id, 3GPP section_id, spec_number).
  - Powered by `WeaviateAdapter` (`app/api/vector_db.py`).

### 4. AI & Machine Learning Layer
- **LLM Engine (`app/core/llm.py`)**: Custom wrapper for the **Groq API**.
  - *Primary*: `llama-3.3-70b-versatile` (Fast, highly capable).
  - *Fallback*: `llama-3.1-8b-instant` (Triggered on rate limits or overload).
  - *Features*: Exponential backoff, deterministic settings (`temperature=0.1`).
- **Embeddings**: **SentenceTransformers** (`BAAI/bge-large-en-v1.5`). 1024-dim, top-tier MTEB retrieval benchmark.
- **Re-ranker (`app/services/reranker.py`)**: **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-12-v2`). Scores query-document pairs jointly for deep token-level relevance.
- **Audio/Speech**: **OpenAI Whisper** (`base` model, thread-safe parallel processing).
- **OCR Engine**: **Tesseract OCR** with `pdf2image` (Poppler).

---

## ⚙️ Core Processing Pipelines

### A. Document Ingestion Pipeline (`file_processing_service.py`)
When a file is uploaded (PDF, DOCX, TXT, Excel, Audio, Image):
1. **Extraction**: Detects format and extracts raw text. (Uses direct PDF reading, falls back to OCR if sparse. Reads Excel into contextual row strings. Transcribes audio).
2. **Chunking (`chunking.py`)**: 
   - *General*: Standard recursive character splitting.
   - *3GPP-Aware*: Detects telecom specs (TS/TR). Splits precisely at section headers (e.g., "4.2.3.1"), preserving hierarchy and injecting the parent section context into *every* chunk.
3. **Embedding**: Generates BGE embeddings in batch.
4. **Indexing**: Upserts into Weaviate. Routes to specific collections based on the user's role (e.g., `telecom_docs`, `finance_docs`).

### B. Retrieval-Augmented Generation (RAG) Pipeline (`rag_service.py`)
When a user asks a question:
1. **Hybrid Retrieval**: 
   - Vector Search (Semantic meaning).
   - BM25 Keyword Search (Exact match for acronyms like UPF, SMF).
2. **Merging & Window Expansion**: Deduplicates results. Expands context by pulling adjacent chunks (±2) to provide the LLM with continuous thought flow.
3. **Cross-Encoder Re-ranking**: Passes expanded chunks through the Cross-Encoder. Discards irrelevant chunks (score < threshold) and sorts the best to the top.
4. **Prompt Engineering**: The `StrategyFactory` injects role-specific rules (e.g., a 3GPP Glossary for engineers).
5. **Generation**: Groq LLM generates the answer, strictly adhering to `[Source X]` citations.

### C. Validation & Confidence Scoring
1. **Citation Enforcer (`citation_enforcer.py`)**: Validates that generated `[Source X]` tags actually exist in the retrieved context and that the cited text semantically supports the claim.
2. **Hallucination Guard (`hallucination_guard.py`)**: Extracts standalone claims from the LLM's answer and verifies them against the source texts using semantic similarity.
3. **Confidence Calculator (`confidence_calculator.py`)**: Combines Faithfulness, Retrieval Quality, Citation Coverage, and Coherence into a final 0-100 score and Hallucination Risk category.

---

## 📂 Directory Structure

```text
Document-AI-Hub/
├── alembic/                 # Database migration scripts
├── app/                     # Backend FastAPI code
│   ├── api/                 # API Endpoints (auth, upload, rag, search, etc.)
│   ├── core/                # Configuration, Security, Custom LLM wrapper
│   ├── db/                  # SQLAlchemy sessions and Base class
│   ├── evaluation/          # Automated RAG test suite
│   ├── generation/          # Post-generation tools (Citation enforcer)
│   ├── metrics/             # Scoring tools (Confidence calculator)
│   ├── models/              # SQLAlchemy Database Models (User, Document, etc.)
│   ├── services/            # Core business logic (Chunking, OCR, RAG Pipeline)
│   └── main.py              # FastAPI Application Entry Point
├── data/                    # Local storage (User uploads, OCR temp files)
├── frontend/                # Streamlit UI code
│   ├── components/          # UI sections (chat, auth, sidebar)
│   ├── utils/               # Frontend helpers (cookie/session manager)
│   ├── api_client.py        # HTTP wrapper for backend
│   └── main.py              # Streamlit Entry Point
├── scripts/                 # Utility scripts (Bulk ingestion, DB fixes)
├── .env                     # Environment variables
├── alembic.ini              # Alembic config
├── docker-compose.yml       # Weaviate configuration
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🚀 Setup & Installation Guide

### Prerequisites
1. **Python 3.11+**
2. **Docker & Docker Compose** (Required for Weaviate Vector DB)
3. **PostgreSQL** (Running locally on default port 5432, or remotely)
4. **Tesseract OCR**: Download and install. Add to system PATH or note the executable path.
5. **Poppler**: Download (for Windows), extract, and note the `\bin` path.
6. **FFmpeg**: Required for audio transcription (Whisper). Must be in system PATH.

### 1. Clone & Environment Setup
Clone the repository and set up a Python virtual environment:
```bash
git clone https://github.com/Brocoder07/Document-AI-Hub.git
cd Document-AI-Hub
python -m venv venv
# Activate virtual environment
.\venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Linux/Mac
# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables (`.env`)
Create a `.env` file in the root directory. Configure paths according to your system:
```ini
# PostgreSQL Connection (Update user/password/db as needed)
DATABASE_URL=postgresql://ai_user:password@localhost/document_ai_hub

# Security (Generate random hashes for these in production)
SECRET_KEY=<your-secure-jwt-secret>
MESSAGE_ENCRYPTION_KEY=<your-fernet-encryption-key>

# Groq API Settings
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MODEL_FALLBACK=llama-3.1-8b-instant

# File Storage Paths (Relative to project root)
UPLOAD_DIR=data/documents
OCR_TEMP_DIR=data/ocr_temp

# System Paths (Windows example. Update for Linux/Mac)
TESSERACT_PATH=C:/Program Files/Tesseract-OCR/tesseract.exe
POPPLER_PATH=C:\Program Files\poppler-24.02.0\Library\bin
```

### 3. Database Initialization
Ensure your PostgreSQL server is running and the database (`document_ai_hub`) exists. Apply the schema:
```bash
alembic upgrade head
```

### 4. Start Weaviate (Vector DB)
Ensure Docker Desktop is running, then spin up Weaviate:
```bash
docker-compose up -d
```

---

## 🏃‍♂️ Running the Application

The application requires both the backend and frontend to run simultaneously. Open two separate terminals.

**Terminal 1 (Backend API):**
```bash
.\venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```
*The API will be available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.*

**Terminal 2 (Frontend UI):**
```bash
.\venv\Scripts\activate
streamlit run frontend/main.py
```
*The UI will automatically open in your browser at `http://localhost:8501`.*

---

## 📚 Advanced Usage Guide

### 1. Document Ingestion (3GPP Specs)
Because telecom standards like TS 23.501 are extremely large (500+ pages), uploading them through the web UI can cause HTTP timeouts. The project includes specialized scripts to handle bulk ingestion safely.

**Direct Bulk Ingestion:**
Place your `.docx`, `.pdf`, or `.txt` 3GPP specifications into `data/documents/` and run the ingest script. This script bypasses the frontend, uses batched embedding to prevent memory crashes, skips boilerplate sections (Annexes, Change History), and injects data directly into Weaviate.
```bash
python scripts/ingest_3gpp.py --dir data/documents
```

### 2. Using the RAG Chat
1. **Register**: Open the frontend. When registering, select **Engineer**, **Telecom Engineer**, or **Network Engineer** as your role. This is critical, as it activates the `TelecomStrategy` in the backend, injecting 3GPP glossaries into the prompt.
2. **Upload Docs**: Navigate to the Document Library. Upload your reference documents.
3. **Chat**: Go to the Chat interface. Ask complex technical questions.
4. **Metrics Tab**: Click the metrics expander below an AI response to view:
   - **Retrieved Documents**: The exact chunks retrieved from Weaviate.
   - **Hallucination Guard**: Claim-by-claim verification (Supported vs. Unsupported).
   - **Confidence Score**: The aggregated metric (0-100%).
   - **Performance**: Retrieval time, Generation time, Token usage.

### 3. Automated Evaluation Suite
To objectively measure the RAG pipeline's accuracy and hallucination rates, run the automated test suite. It asks a series of predefined questions (accuracy, adversarial, and refusal tests) and generates a detailed JSON report.
```bash
python -m app.evaluation.run_eval --user-id 1
```
*(Replace `1` with the database ID of your registered user).*

---

## 🛠 Troubleshooting

- **Database Errors on Startup**: Ensure PostgreSQL is running and the credentials in `.env` are completely accurate. If you altered models, run `alembic revision --autogenerate -m "msg"` followed by `alembic upgrade head`.
- **`WeaviateConnectionError`**: Docker is likely not running, or the container failed to start. Run `docker-compose ps` to check.
- **`UnicodeEncodeError` in Terminal (Windows)**: Windows terminals default to `cp1252` encoding. The codebase has been sanitized of emojis to prevent crashes, but if you run custom scripts and see this error, ensure your terminal supports UTF-8.
- **Missing `torch` / SentenceTransformers errors**: Ensure you are using the exact pinned versions in `requirements.txt`. Newer versions may have conflicting dependencies.
- **Audio Transcription Fails**: Ensure `ffmpeg` is installed and correctly added to your system's PATH. Use `ffmpeg -version` in the terminal to verify.
