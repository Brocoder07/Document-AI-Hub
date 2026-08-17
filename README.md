# Document AI Hub

Document AI Hub is a production-grade Retrieval-Augmented Generation (RAG) application specifically designed to process, index, and query massive telecommunications standards documentation (e.g., 3GPP TS and TR specifications) with a focus on **minimal to near-zero hallucinations**. 

The system provides role-based access, supporting various specialized domains (Telecom, Legal, Healthcare, Finance, Academic, Business) via a Strategy Pattern, with telecom being the primary focus.

## 🏗 System Architecture

The application is built on a modern, decoupled architecture:

1. **Frontend (User Interface)**: Built with **Streamlit**. Provides user authentication, a document library for file uploads, and an interactive chat interface that displays not just the AI's answer, but also retrieved context, processing metrics, confidence scores, and hallucination risks.
2. **Backend (API Server)**: Built with **FastAPI**. Exposes RESTful endpoints for authentication, file upload processing, OCR, vector search, and the core RAG pipeline.
3. **Relational Database**: **PostgreSQL** managed via **SQLAlchemy** (ORM) and **Alembic** (migrations). Stores user accounts, chat session histories, and document tracking metadata.
4. **Vector Database**: **Weaviate** running in a local Docker container. Stores document chunks, rich metadata (including 3GPP section numbers), and embeddings for hybrid search.
5. **LLM Engine**: Powered by the **Groq API** (using `llama-3.3-70b-versatile` with an automatic fallback to `llama-3.1-8b-instant`). Features custom retry logic with exponential backoff for rate limiting.
6. **Embedding Model**: Local inference using **SentenceTransformers** (`BAAI/bge-large-en-v1.5`).
7. **Re-ranker**: Local inference using **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-12-v2`).

## ✨ Key Technical Features

- **Hybrid Retrieval**: Combines semantic vector similarity search with BM25 keyword search to ensure exact-match acronyms (like AMF, SMF, UPF) are not missed.
- **3GPP-Aware Chunking**: Intelligently splits massive 3GPP `.docx` and `.pdf` files while preserving section hierarchies (e.g., "4.2.3.1") and injecting parent context into every chunk.
- **Post-Generation Hallucination Guard**: A custom module (`hallucination_guard.py`) that extracts individual claims from the LLM's answer, verifies them against the retrieved source texts using semantic similarity, and calculates a strict faithfulness score.
- **Domain-Specific Strategies**: Injects specific system prompts and glossaries based on the user's role (e.g., the `TelecomStrategy` injects a 3GPP glossary to clarify acronyms).
- **Direct Ingestion Pipeline**: Dedicated scripts to handle massive 500+ page telecom documents locally without overwhelming the frontend.

---

## 🚀 Setup & Installation Guide

### Prerequisites
- **Python 3.11+**
- **Docker & Docker Compose** (for Weaviate)
- **PostgreSQL** (running locally or remotely)
- **Tesseract OCR** (installed on the system, e.g., `C:/Program Files/Tesseract-OCR/tesseract.exe`)
- **Poppler** (for PDF processing, added to system PATH)

### 1. Clone & Environment Setup
Clone the repository and set up a Python virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

### 2. Environment Variables (`.env`)
Create a `.env` file in the root directory (or update the existing one) with the following essential variables:
```ini
DATABASE_URL=postgresql://<user>:<password>@localhost/<dbname>
SECRET_KEY=<your-secure-jwt-secret>
MESSAGE_ENCRYPTION_KEY=<your-fernet-encryption-key>
GROQ_API_KEY=<your-groq-api-key>
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MODEL_FALLBACK=llama-3.1-8b-instant
TESSERACT_PATH=C:/Program Files/Tesseract-OCR/tesseract.exe
```
*(See the existing `.env` file for all default paths and configurations).*

### 3. Database Initialization
Start your PostgreSQL server and ensure the database specified in `DATABASE_URL` exists. Then, run Alembic migrations to create the tables:
```bash
alembic upgrade head
```

### 4. Start Weaviate (Vector DB)
Ensure Docker is running, then spin up the local Weaviate instance:
```bash
docker-compose up -d
```

---

## 🏃‍♂️ Running the Application

You need to run both the FastAPI backend and the Streamlit frontend simultaneously. Open two separate terminals.

**Terminal 1 (Backend):**
```bash
.\venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```bash
.\venv\Scripts\activate
streamlit run frontend/main.py
```
*The frontend will be accessible at `http://localhost:8501`.*

---

## 📚 Document Ingestion (3GPP Specs)

Because telecom standards like TS 23.501 are extremely large (500+ pages, 7MB+ Word docs), uploading them through the web UI can cause timeouts or memory issues. The project includes specialized scripts to handle this.

### Option A: Direct Bulk Ingestion (Recommended)
Place your `.docx` or `.txt` 3GPP specifications into `data/documents/` and run the ingest script. This script bypasses the frontend, uses batch embedding (50 chunks at a time) to prevent memory crashes, and skips boilerplate sections (Annexes, Change History).
```bash
python scripts/ingest_3gpp.py --dir data/documents
```

### Option B: Split & Upload via UI
If you prefer to test the frontend upload functionality, you can use the splitter script. This reads massive `.docx` files and extracts the core technical content into smaller, ~100KB `.txt` files.
```bash
python scripts/split_3gpp_v2.py
```
You can then log into the frontend, navigate to **Document Library**, and upload these smaller `.txt` files manually.

---

## 🧑‍💻 Usage Guide

1. **Register an Account**: Open the frontend and create an account. **Crucial**: Select `Engineer`, `Telecom Engineer`, or `Network Engineer` as your role to trigger the `TelecomStrategy` in the RAG pipeline.
2. **Upload/Index Documents**: Ensure documents are indexed either via the web UI or the ingestion scripts.
3. **Chat**: Navigate to the Chat interface. Ask complex telecom questions (e.g., *"What is the procedure for UE-triggered Service Request?"*).
4. **Review Metrics**: The chat UI includes expandable metric tabs for each answer:
   - **Retrieved Documents**: See the exact chunks used.
   - **Hallucination Guard**: View the claim-by-claim verification breakdown (Supported vs. Unsupported).
   - **Confidence Score**: View the overall faithfulness percentage.
   - **Token Usage & Latency**: Monitor Groq API performance.

## 🛠 Troubleshooting

- **UnicodeEncodeError (`charmap` codec can't encode character)**: Windows terminals default to `cp1252` which crashes on emojis. The codebase has been sanitized of emojis in log statements, but if you run custom scripts, ensure you add `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` at the top of your scripts.
- **Weaviate Connection Issues**: Ensure Docker is running and ports `8080` and `50051` are available.
- **Missing `torch` / SentenceTransformers errors**: Ensure you are using the pinned versions in `requirements.txt` (`transformers==4.39.3`, `sentence-transformers==2.7.0`). Newer versions have conflicting dependencies with `torch`.
