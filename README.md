# Multimodal Offline RAG System
### SIH25231 — Offline Multimodal Intelligence Assistant

Fully offline AI knowledge assistant. No internet required during inference.  
Supports: **PDF · DOCX · Images (OCR) · Audio/Voice (Whisper STT)**

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        UI (Gradio)                       │
│              http://localhost:7860                       │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────────┐
│                  FastAPI Backend                         │
│               http://localhost:8000                     │
│                                                         │
│  /upload → ingest.py → vectorstore.py                  │
│  /query  → vectorstore.py → llm.py → response          │
└──────┬────────────────────────┬────────────────────────┘
       │                        │
┌──────▼──────┐     ┌───────────▼───────────────────────┐
│  ChromaDB   │     │         Ollama                    │
│ (vectordb/) │     │   localhost:11434                 │
│ + MiniLM    │     │   llama3.2:3b (4GB VRAM)         │
│  embeddings │     └───────────────────────────────────┘
└─────────────┘
```

## File Processing Pipeline

```
File Upload
    │
    ├── .pdf  → PyMuPDF (native text) → Tesseract OCR (scanned pages)
    ├── .docx → python-docx (paragraphs + tables)
    ├── .png/.jpg/.jpeg/.tiff → Tesseract OCR
    └── .mp3/.wav/.mp4/.m4a → Whisper base (offline STT)
    │
    ▼
Text Chunks (512 tokens, 64 overlap)
    │
    ▼
Embeddings (all-MiniLM-L6-v2, CPU)
    │
    ▼
ChromaDB (persistent, cosine similarity)
```

## Hardware Recommendations

| Component | Min | Used Here |
|-----------|-----|-----------|
| GPU VRAM | 4GB | RTX 3050 4GB |
| RAM | 8GB | Manageable |
| Storage | 10GB free | SSD preferred |
| OS | Linux/Windows/Mac | Any |

## Setup (First Time)

```bash
# Clone or extract the project
cd multimodal-rag

# Run setup (installs deps, Tesseract, pulls LLM)
bash setup.sh
```

## Running

Open **3 separate terminals**:

```bash
# Terminal 1 — Ollama LLM server
ollama serve

# Terminal 2 — API backend
python api.py
# → http://localhost:8000/docs  (Swagger UI)

# Terminal 3 — Gradio frontend
python ui.py
# → http://localhost:7860
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System status |
| `POST` | `/upload` | Upload & index a file |
| `POST` | `/query` | Ask a question |
| `GET` | `/documents` | List indexed files |
| `DELETE` | `/documents/{filename}` | Remove a file |
| `GET` | `/stats` | Vector store stats |

## Switching LLM Models

Edit `llm.py` → `DEFAULT_MODEL`:

```python
DEFAULT_MODEL = "llama3.2:3b"    # Default — 4GB VRAM
# DEFAULT_MODEL = "phi3.5"       # Smaller, faster
# DEFAULT_MODEL = "mistral:7b-q4" # Better quality, needs 6GB VRAM
```

Then pull with: `ollama pull <model-name>`

## Project Structure

```
multimodal-rag/
├── ingest.py        # File extraction (PDF/DOCX/Image/Audio)
├── vectorstore.py   # ChromaDB operations + embeddings
├── llm.py           # Ollama LLM wrapper + RAG prompt
├── api.py           # FastAPI REST API
├── ui.py            # Gradio frontend
├── requirements.txt
├── setup.sh
├── uploads/         # Uploaded files (auto-created)
└── vectordb/        # ChromaDB persistent store (auto-created)
```

## Troubleshooting

**Ollama not connecting**  
→ Run `ollama serve` in a separate terminal

**Out of VRAM**  
→ Ollama auto-offloads to CPU. Set `"num_gpu": 0` in `llm.py` options to force CPU.

**Tesseract not found**  
→ Windows: install from https://github.com/UB-Mannheim/tesseract/wiki, add to PATH

**Audio transcription slow**  
→ Change `whisper.load_model("base")` to `"tiny"` in `ingest.py` for 3x speed

**8GB RAM pressure**  
→ Don't upload audio and large PDFs simultaneously. Process sequentially.
