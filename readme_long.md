# Research Assistant (Hybrid RAG, Local, Ollama)

A fully local, hybrid Retrieval-Augmented Generation (RAG) research assistant built with:

* **LangChain**
* **Ollama**
* **FAISS**
* **BM25 hybrid retrieval**
* Session memory
* Config-driven architecture
* Persistent logging
* Corpus change detection

Designed for serious document research with transparent citations.

---

## Features

* Hybrid retrieval (Vector + BM25)
* Metadata-aware citations (title, section, page)
* Session-based conversational memory
* Configurable via `config.json`
* Command-line overrides
* Automatic FAISS rebuild on corpus change
* Optional conversation logging
* Optional retrieval logging

---

## Architecture

### LLM

Uses `ChatOllama`

Default model:

```
qwen3:4b-q4_K_M
```

Temperature configurable in config.

---

### Embeddings

Uses `OllamaEmbeddings`

Default:

```
nomic-embed-text
```

---

### Vector Store

* FAISS
* Local disk persistence
* Automatic rebuild when corpus changes
* Manifest-based change detection

---

### Hybrid Retrieval

1. FAISS semantic retrieval (`vector_k`)
2. BM25 lexical retrieval (`bm25_k`)
3. Results merged + deduplicated
4. Metadata preserved

---

### Supported Document Types

| Type         | Notes                          |
| ------------ | ------------------------------ |
| `.txt`       | First line used as title       |
| `.pdf`       | Page numbers preserved         |
| `.docx`      | Section detection via headings |
| `.odt`       | Supported                      |
| `.html/.htm` | Section extraction from h1h3  |

Metadata stored per chunk:

* title
* source filename
* section
* page
* file_type

---

## Installation

### 1. Install Ollama

[https://ollama.com](https://ollama.com)

Pull required models:

```
ollama pull qwen3:4b-q4_K_M
ollama pull nomic-embed-text
```

---

### 2. Install Python Dependencies

```
pip install langchain langchain-core langchain-community \
            langchain-ollama faiss-cpu \
            unstructured python-docx beautifulsoup4 lxml
```

(You may adjust based on your environment.)

---

## Usage

Basic run:

```
python research_assistant_v6.py
```

---

### Command Line Options

```
usage: research_assistant.py [-h] [-c CONFIG] [-m MODEL] [-d DOCUMENTS]
```

Options:

| Flag         | Description               |
| ------------ | ------------------------- |
| `-h, --help` | Show help message         |
| `-c`         | Path to config file       |
| `-m`         | Override LLM model        |
| `-d`         | Override papers directory |

---

### Examples

Use custom config:

```
python research_assistant_v6.py -c research_config.json
```

Change model:

```
python research_assistant_v6.py -m llama3.1
```

Change corpus directory:

```
python research_assistant_v6.py -d ./religion/BofConcord
```

Combine overrides:

```
python research_assistant_v6.py -c prod.json -m qwen3:8b -d ./legal_docs
```

---

## Configuration

Uses `config.json`.

If missing, defaults are used.

Example:

```json
{
  "paths": {
    "papers_dir": "./papers",
    "vectorstore_dir": "./faiss_index",
    "conversation_dir": "./conversations",
    "retrieval_log_dir": "./retrieval_logs"
  },
  "models": {
    "embedding_model": "nomic-embed-text",
    "chat_model": "qwen3:4b-q4_K_M",
    "temperature": 0.2
  },
  "retrieval": {
    "vector_k": 4,
    "bm25_k": 4
  },
  "chunking": {
    "chunk_size": 800,
    "chunk_overlap": 150
  },
  "logging": {
    "log_conversations": false,
    "log_retrieval_metadata": false
  },
  "session": {
    "default_session_id": "default"
  }
}
```

Precedence order:

```
CLI arguments
> config.json
> DEFAULT_CONFIG
```

---

## Index Rebuild Logic

The system maintains a `manifest.json` in the FAISS directory.

On startup:

* If manifest missing → rebuild
* If file size or mtime changed → rebuild
* If unchanged → load existing index

BM25 always reloads documents.

---

## Logging

### Conversation Logging (Optional)

When enabled:

* Stored per session as JSON
* Includes:

  * timestamp
  * model
  * question
  * answer

---

### Retrieval Logging (Optional)

Logs:

* Query
* Retrieved document metadata
* Timestamp

Useful for evaluation and debugging.

---

## Citation Format

The assistant cites using:

```
(Title  source, section: X, page: Y)
```

At the end of each answer:

```
Sources:
- ...
```

---

## Example Workflow

1. Place documents in `./papers`
2. Run the assistant
3. Choose a session name
4. Ask research questions
5. Receive cited answers grounded in your corpus

---

## Current Version

v8

* Hybrid retrieval
* Session memory
* Logging
* Config file
* Command-line arguments

---

## Planned Improvements

* Weighted hybrid scoring
* Reranking
* Chunk caching
* Streaming output
* Evaluation mode
* Class-based architecture
* API interface

---

## Philosophy

* Fully local
* Transparent citations
* Configurable
* Research-focused
* No external API dependency

---

If you’d like, I can now:

* Rewrite this as a shorter README (cleaner for GitHub)
* Write a more technical architecture README
* Add a “Screenshots” section template
* Add a “Contributing” section
* Add license suggestions
* Or turn this into a polished project landing README

