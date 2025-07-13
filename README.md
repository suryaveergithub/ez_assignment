# ez_assignment
# DocSmart AI

DocSmart AI is a research assistant web application that allows users to:

- Upload PDF or TXT documents
- Automatically generate concise summaries
- Ask questions based on document content ("Ask Anything" mode)
- Take an auto-generated logic quiz based on the uploaded text ("Challenge Me" mode)

Built using Python, Streamlit, and lightweight NLP libraries to ensure fast, offline-compatible performance.

---

## Features

### 1. Document Upload
Supports `.pdf` and `.txt` formats. Extracts all readable text using PyMuPDF or standard decoding.

### 2. Automatic Summarization
Uses SpaCy-based extractive summarization. Clean, paragraph-style output optimized for human readability.

### 3. Ask Anything Mode
Embeds document content using SentenceTransformers and retrieves the most relevant answer using FAISS similarity search.

### 4. Challenge Me Mode
Generates logic-based fill-in-the-blank questions. The user's answers are semantically evaluated with similarity scoring.

---

## Tech Stack

- Python 3.12
- Streamlit
- SpaCy
- SentenceTransformers (MiniLM)
- FAISS (CPU)
- PyMuPDF (PDF parsing)
