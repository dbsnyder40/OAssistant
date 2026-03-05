# OAssistant.py - Research Assistant (Local Hybrid RAG)

A **fully local Retrieval-Augmented Generation (RAG) research assistant** built with:

* LangChain 1.x
* Ollama
* FAISS
* Hybrid retrieval (Vector + BM25)

The system is designed for **document-driven research with transparent citations, structured metadata, and zero external APIs**.

---

# Features

* **Hybrid retrieval**

  * Semantic search (FAISS embeddings)
  * Lexical search (BM25)
* **Structured document ingestion**
* **Metadata-rich document chunks**
* **Automatic FAISS rebuild when corpus changes**
* **Section-aware document parsing**
* **Session-based conversational memory**
* **Config-driven architecture**
* **Optional conversation & retrieval logging**
* **Command-line overrides**

---

# Requirements

* Python **3.10+**
* **Ollama** installed locally
  [https://ollama.com](https://ollama.com)

Pull the required models:

```bash
ollama pull qwen3:4b-q4_K_M
ollama pull nomic-embed-text
```

Install dependencies:

```bash
pip install langchain langchain-core langchain-community \
            langchain-ollama faiss-cpu \
            python-docx beautifulsoup4 lxml \
            markdown-it-py
```

Optional but recommended:

```bash
sudo apt install pandoc
```

Pandoc is used to convert **ODT → DOCX** during ingestion.

---

# Usage

Place documents in the configured `papers` directory and run:

```bash
python oassistant.py
```

The system will:

1. Load documents
2. Chunk them
3. Create embeddings
4. Build or update the FAISS index
5. Start an interactive research chat

---

# Command Line Options

```
python research_assistant_v6.py [-h] [-c CONFIG] [-m MODEL] [-d DOCUMENTS]
```

| Flag | Description                 |
| ---- | --------------------------- |
| `-h` | Show help                   |
| `-c` | Path to configuration file  |
| `-m` | Override LLM model          |
| `-d` | Override document directory |

Example:

```bash
python research_assistant_v6.py -m llama3.1 -d ./religion
```

CLI arguments override `config.json`, which overrides defaults.

---

# Supported File Types

The ingestion pipeline uses **format-specific loaders** designed for consistent metadata.

| Format  | Strategy                        |
| ------- | ------------------------------- |
| `.txt`  | First line used as title        |
| `.pdf`  | Page-based loading              |
| `.docx` | Section detection via headings  |
| `.odt`  | Converted to DOCX via Pandoc    |
| `.md`   | Parsed via Markdown AST         |
| `.html` | Section extraction via headings |
| `.json` | Chat conversation memory        |

---

# Document Metadata

Every chunk stores structured metadata used during retrieval:

| Field       | Meaning                                   |
| ----------- | ----------------------------------------- |
| `source`    | Original filename                         |
| `file_type` | txt / pdf / docx / odt / md / html / json |
| `title`     | Extracted or inferred document title      |
| `section`   | Section heading if available              |
| `page`      | Page number (PDF only)                    |
| `corpus`    | Document collection identifier            |

Optional metadata may include:

* `contains_code`
* `code_language`
* `session_id`

This metadata improves **citation accuracy and retrieval filtering**.

---

# Document Ingestion Pipeline

Documents are processed using a **deterministic loader architecture** rather than generic “magic” loaders.

Key design decisions:

* Avoid `Unstructured` loaders where possible
* Use **format-specific parsers**
* Preserve **section hierarchy**
* Attach **consistent metadata**

### ODT Handling

ODT files are converted using:

```
ODT → DOCX → parsed via python-docx
```

This avoids unreliable `UnstructuredODTLoader` behavior.

---

# How Retrieval Works

1. Documents are loaded and converted to `Document` objects.

2. Documents are chunked.

3. Embeddings are generated with Ollama.

4. FAISS stores vectors locally.

5. Hybrid retrieval merges:

   * Semantic similarity (FAISS)
   * Lexical matches (BM25)

6. Retrieved context is passed to the LLM.

7. The response includes **transparent citations**.

---

# Index Rebuild Logic

A **file manifest** tracks:

* filenames
* modification timestamps

If the corpus changes, the system automatically:

```
rebuilds the FAISS index
```

This prevents stale embeddings.

---

# Configuration

If `config.json` exists, it overrides default settings.

You can configure:

* LLM model
* Embedding model
* Retrieval parameters
* Chunk size
* Logging
* Document paths

---

# Project Goals

* Fully local research assistant
* High-quality document retrieval
* Transparent citations
* Structured document ingestion
* Extensible architecture
