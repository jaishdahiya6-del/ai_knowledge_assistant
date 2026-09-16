# 🧠 AI Personal Knowledge & Research Assistant

A self-contained, **fully local** AI dashboard built with **Python, Pandas, NumPy,
Scikit-learn, PyTorch, NLP, and Streamlit**. Track what you learn, explore it with
interactive EDA, predict priorities with a Random Forest and a PyTorch neural
network, run NLP text analysis, and chat with a local AI assistant that answers
questions using **your own stored data**.

**No paid API keys. No external file uploads. No backend server.** Everything runs
on your machine and is stored in a local SQLite database.

---

## ✨ Features

| Area | What it does |
|---|---|
| **Data Entry** | In-app form to log learning entries (title, category, notes, priority, hours) |
| **Sample Data** | One-click generator creates 20–300 realistic synthetic entries |
| **Storage** | SQLite database (`data/knowledge.db`), created automatically |
| **EDA** | Pandas/NumPy summary stats, category breakdowns, correlation heatmap |
| **Charts** | Interactive Plotly charts (bar, line, scatter, heatmap) |
| **ML Prediction** | Scikit-learn Random Forest classifier + Linear Regression, with accuracy, classification report, confusion matrix, feature importances |
| **Deep Learning** | A PyTorch feed-forward neural network (trained live, loss curves, live predictions) |
| **NLP** | Tokenization, word frequency, TF-IDF keyword extraction, lexicon-based sentiment analysis — all offline |
| **AI Assistant** | Local TF-IDF + cosine-similarity retrieval chatbot that answers questions from your own notes, plus rule-based stats queries |
| **Data Management** | Export to CSV, clear database, regenerate sample data |

---

## 📁 Project Structure

```
ai_knowledge_assistant/
├── app.py                  # Main Streamlit application (entry point)
├── requirements.txt
├── README.md
├── data/                   # SQLite DB lives here (auto-created)
└── src/
    ├── __init__.py
    ├── database.py         # SQLite read/write layer
    ├── data_generator.py   # Synthetic sample data generator
    ├── eda.py               # Pandas/NumPy/Plotly EDA functions
    ├── ml_models.py         # Scikit-learn classifier + regressor
    ├── neural_net.py        # PyTorch neural network
    ├── nlp_analysis.py      # Tokenization, TF-IDF, sentiment
    └── assistant.py         # Local retrieval-based AI assistant
```

---

## 🖥️ Setup — Windows (CMD)

```cmd
:: 1. Clone or extract the project, then open the folder
cd ai_knowledge_assistant

:: 2. Create a virtual environment
python -m venv venv

:: 3. Activate it
venv\Scripts\activate

:: 4. Install dependencies
pip install -r requirements.txt

:: 5. Run the app
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**.

---

## 🖥️ Setup — VS Code

1. Open the `ai_knowledge_assistant` folder in VS Code (`File > Open Folder`).
2. Open a terminal: `` Ctrl+` ``.
3. Create & activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```
4. Select the venv interpreter: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv`/`venv`.
5. Install dependencies: `pip install -r requirements.txt`.
6. Run: `streamlit run app.py` (or use the VS Code "Run" button on `app.py` if you prefer a task).

---

## 🚀 Quick Start

1. Launch the app (`streamlit run app.py`).
2. Go to **⚙️ Data Management** → click **Generate Sample Data** (creates 120 entries by default).
3. Explore **📊 EDA & Charts** for visual analysis.
4. Go to **🤖 ML Prediction** → click **Train / Retrain Models**.
5. Go to **🔥 Neural Network** → click **Train Neural Network**.
6. Go to **📝 NLP Analysis** to see keyword extraction & sentiment.
7. Go to **💬 AI Assistant** and ask questions like:
   - "How many entries do I have?"
   - "What category do I spend the most time on?"
   - "What should I study next?"
   - "Tell me about gradient descent" (retrieves relevant notes)
8. Add your own real notes anytime via **✍️ Add Entry**.

---

## 🧩 Tech Notes

- **ML models require ≥20 entries** to train reliably (enforced with a friendly warning in the UI). Generate sample data if you don't have enough entries yet.
- The **AI Assistant** uses TF-IDF vectorization + cosine similarity (Scikit-learn) over your notes — a legitimate, explainable retrieval technique that requires no internet connection or paid API.
- The **NLP sentiment analyzer** uses a small built-in positive/negative word lexicon so it works fully offline (no `nltk.download` needed).
- All charts are **Plotly** (interactive: zoom, pan, hover) rendered inside Streamlit; some quick views use Streamlit's native `st.line_chart`/`st.bar_chart`.
- Error handling wraps all DB writes, model training, and predictions so the app degrades gracefully instead of crashing.

---

## 🛠️ Troubleshooting

- **"Need at least 20 entries"** → Generate sample data from ⚙️ Data Management.
- **PyTorch install issues on Windows** → Use the CPU wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- **`pandas` fails to build from source (`metadata-generation-failed`, `Could not find ... vswhere.exe`)** → This happens when pip can't find a prebuilt wheel for your Python version and tries to compile pandas from source, which needs Visual Studio Build Tools. Fix it one of two ways:
  1. **Recommended:** Use Python 3.11 or 3.12 (prebuilt wheels are readily available), re-create the venv, and re-run `pip install -r requirements.txt`.
  2. Or install the **"Desktop development with C++"** workload from the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) so pip can compile it.
  `requirements.txt` now uses `>=` version bounds instead of exact pins, so pip will automatically grab the newest compatible wheel for your Python version.
- **Port already in use** → `streamlit run app.py --server.port 8502`.
- **Database locked errors** → Close other running instances of the app pointing at the same `data/knowledge.db`.

---

## 📜 License

Free to use, modify, and extend for personal or educational purposes.
