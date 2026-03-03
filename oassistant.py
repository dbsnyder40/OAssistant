#!/usr/bin/env python3
# oassistant.py

import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

import json
import copy
import argparse
import shutil


from oassistant.indexing.manifest import needs_rebuild, build_file_manifest
from oassistant.indexing.loaders import load_documents
from oassistant.indexing.chunking import chunk_documents
from oassistant.indexing.vectorstore import build_vectorstore
from oassistant.chat.rag_chain import create_rag_chain
from oassistant.chat.logging import log_retrieval, save_conversation
from oassistant.chat.session import interactive_session
from oassistant.models import create_embeddings, create_chat_model
from oassistant.version import VERSION

# =============================
# Configuration
# =============================

DEFAULT_CONFIG = {
    "paths": {
        "papers_dir": "./papers",
        "vectorstore_dir": "./faiss_index",
        "conversation_dir": "./conversations",
        "retrieval_log_dir": "./retrieval_logs"
    },
    "models": {
        "provider":"ollama",
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
        "log_conversations": False,
        "log_retrieval_metadata": False
    },
    "session": {
        "default_session_id": "default"
    },
    "corpus":"research"
}

def load_config(path="config.json"):
    if not Path(path).exists():
        print("No config.json found. Using defaults.")
        return copy.deepcopy(DEFAULT_CONFIG)

    with open(path) as f:
        user_config = json.load(f)

    return deep_merge(copy.deepcopy(DEFAULT_CONFIG), user_config)
            
def deep_merge(default, override):
    for key, value in override.items():
        if (
            key in default
            and isinstance(default[key], dict)
            and isinstance(value, dict)
        ):
            default[key] = deep_merge(default[key], value)
        else:
            default[key] = value
    return default


def parse_args():
    parser = argparse.ArgumentParser(
        prog="oassistant.py",
        description="Local Hybrid RAG Research Assistant using LangChain + Ollama.",
        epilog="CLI overrides config.json. Config overrides defaults.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to configuration file"
    )

    parser.add_argument(
        "-m", "--model",
        help="Override LLM chat model (e.g. qwen3:4b-q4_K_M)"
    )

    parser.add_argument(
        "-d", "--documents",
        help="Override papers directory"
    ) 
    
    # ✅ Version flag
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    return parser.parse_args() 

# =============================
# Main
# =============================

if __name__ == "__main__":

    args = parse_args()

    # Load config file (or defaults)
    config = load_config(args.config)

    # =============================
    # CLI Overrides (highest priority)
    # =============================

    if args.model:
        config["models"]["chat_model"] = args.model

    if args.documents:
        config["paths"]["papers_dir"] = args.documents

    # Optional: show effective config
    print("\nEffective configuration:")
    print(json.dumps(config, indent=2))

    papers_dir = Path(config["paths"]["papers_dir"])
    db_dir = Path(config["paths"]["vectorstore_dir"])
    
    corpus_name = config["corpus"]

    if needs_rebuild(config):
        print("Corpus changed  rebuilding index.")

        documents = load_documents(papers_dir, corpus_name)
        chunks = chunk_documents(documents, config)

        # delete old index if exists
        if db_dir.exists():

            shutil.rmtree(db_dir)


        # rebuid manifest
        print("rebuilding vectorstore ...")
        db_dir.mkdir(exist_ok=True)
        vectorstore = build_vectorstore(chunks, config)
        manifest = {
            "corpus": corpus_name,
            "files": build_file_manifest(papers_dir)
        }

        db_dir.mkdir(exist_ok=True)         
        with open(db_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    else:
        print("Corpus unchanged  loading existing index.")
        embeddings = create_embeddings(config)
        vectorstore = FAISS.load_local(
            str(db_dir),
            embeddings,
            allow_dangerous_deserialization=True
        )

        # 🔹 Still reload docs for BM25
        documents = load_documents(papers_dir, corpus_name)
        chunks = chunk_documents(documents,config)
 
    rag_chain = create_rag_chain(vectorstore, chunks, config)
   
    interactive_session(rag_chain,config)
