# 🧠 AI Personal Knowledge & Research Assistant

A self-contained AI Knowledge Assistant built with **Python, SentenceTransformers, FAISS, PyTorch, Scikit-learn, and Streamlit**. Ingest documents (PDF, TXT, Markdown), index them with dense vector embeddings, and answer user questions accurately with explicit source citations.

---

## ✨ Features

| Area | What it does |
|---|---|
| **Document Ingestion** | Load `.pdf`, `.txt`, and `.md` files, extract text with page/source metadata, and split into overlapping chunks |
| **Vector Store** | Fast semantic similarity search using SentenceTransformers (`all-MiniLM-L6-v2`) and FAISS |
| **RAG & Citations** | Context-augmented QA using OpenAI LLM (or offline fallback) with explicit source citations (`[Source: doc.pdf, Page 1, Chunk #0]`) |
| **CLI Tool** | Full command-line interface (`cli.py`) for document ingestion and QA |
| **Web Dashboard** | Interactive Streamlit UI (`app.py`) for document uploads, RAG chat with citation expanders, analytics, ML predictions, and DB management |
| **Configuration** | Environment variables managed via `.env` / `python-dotenv` (API keys never hardcoded) |
| **Unit Tests** | Complete unit test suite in `tests/` covering loaders, vector store, and RAG retrieval |

---

## 📁 Project Structure

```
ai_knowledge_assistant/
├── app.py                  # Main Streamlit web application
├── cli.py                  # Command-line interface for RAG ingestion & ask
├── requirements.txt
├── .env.example            # Environment variables template
├── README.md
├── data/                   # SQLite DB and FAISS vector index live here
├── src/
    ├── __init__.py
    ├── document_loader.py  # PDF, TXT, MD parser and chunker
    ├── embeddings.py       # SentenceTransformers embedding generator
    ├── vector_store.py     # FAISS vector store save/load/search
    ├── rag_assistant.py    # RAG prompt builder, OpenAI LLM integration & citations
    ├── database.py         # SQLite database layer
    ├── data_generator.py   # Synthetic sample data generator
    ├── eda.py              # Exploratory data analysis & Plotly charts
    ├── ml_models.py        # Scikit-learn Random Forest & Linear Regression
    ├── neural_net.py       # PyTorch neural network
    ├── nlp_analysis.py     # TF-IDF & sentiment analysis
    └── assistant.py        # SQLite database notes chatbot
└── tests/
    ├── test_document_loader.py
    ├── test_vector_store.py
    └── test_rag_assistant.py
```

---

## ⚙️ Environment Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. (Optional) Set your OpenAI API key in `.env` if you want LLM-powered answer generation:
   ```env
   OPENAI_API_KEY=your_actual_openai_api_key
   ```
   *Note: If no API key is provided, the assistant operates seamlessly in offline mode using retrieved context excerpts and citations.*

---

## 🖥️ Setup Instructions

```bash
# 1. Clone or navigate to project folder
cd ai_knowledge_assistant

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux / macOS
# or: venv\Scripts\activate # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit web dashboard
streamlit run app.py
```

---

## 💻 CLI Usage

The command-line tool `cli.py` allows ingesting documents and asking questions directly from your terminal:

```bash
# Ingest a directory or file of documents (.pdf, .txt, .md)
python cli.py ingest --path /path/to/documents/

# Ask a question to the assistant
python cli.py ask "What is gradient descent?"

# Customize top-k retrieval count
python cli.py ask "How does random forest work?" --top-k 5
```

---

## 🧪 Running Unit Tests

Run the full automated unit test suite:

```bash
python -m unittest discover tests
```

---

## 📜 License

Free to use, modify, and extend for personal or educational purposes.
