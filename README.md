# OAssistant
Ollama based research assistant
# Research Assistant (Local Hybrid RAG)

A fully local Retrieval-Augmented Generation (RAG) research assistant built with:

* LangChain
* Ollama
* FAISS
* Hybrid retrieval (Vector + BM25)

Designed for document-based research with transparent citations and zero external APIs.

---

## Features

* Hybrid retrieval (semantic + lexical)
* Automatic FAISS rebuild when corpus changes
* Metadata-aware citations (title, section, page)
* Session-based conversational memory
* Config-driven architecture
* Optional conversation & retrieval logging
* Command-line overrides

---

## Requirements

* Python 3.10+
* Ollama installed locally → [https://ollama.com](https://ollama.com)

Pull required models:

```bash
ollama pull qwen3:4b-q4_K_M
ollama pull nomic-embed-text
```

Install dependencies:

```bash
pip install langchain langchain-core langchain-community \
            langchain-ollama faiss-cpu \
            unstructured python-docx beautifulsoup4 lxml
```

---

## Usage

Place documents in the configured `papers` directory, then:

```bash
python oassistant.py
```

---

### Command Line Options

```bash
python oassistant.py [-h] [-c CONFIG] [-m MODEL] [-d DOCUMENTS]
```

| Flag | Description               |
| ---- | ------------------------- |
| `-h` | Show help                 |
| `-c` | Path to config file       |
| `-m` | Override LLM model        |
| `-d` | Override papers directory |

Example:

```bash
python oassistant.py -m llama3.1 -d ./religion
```

CLI overrides `config.json`, which overrides defaults.

---

## Supported File Types

* `.txt` (first line = title)
* `.pdf` (page numbers preserved)
* `.docx` (section detection via headings)
* `.odt`
* `.html`
* '.md'
* '.json' (conversations)

Each chunk stores:

* title
* source filename
* section
* page
* file type

---

## How It Works

1. Documents are loaded and chunked.
2. Embeddings are created with Ollama.
3. FAISS stores vectors locally.
4. Hybrid retrieval merges:

   * FAISS semantic results
   * BM25 lexical results
5. The LLM answers using only retrieved context.
6. Citations are included in every response.

A file manifest detects corpus changes and triggers automatic index rebuild.

---

## Configuration

If `config.json` exists, it overrides defaults.

You can configure:

* Models
* Retrieval parameters
* Chunking size
* Logging
* Paths

---

## Project Goals

* Fully local
* Transparent citations
* Research-focused
* Extensible architecture

