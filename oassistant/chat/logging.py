# oassistant/chat/logging.py

"""
    log_retrieval()
    save_conversation()
"""

import json
from pathlib import Path
from datetime import datetime

from typing import Any, List
from langchain_core.documents import Document

def _safe_load_json(filepath: Path) -> List[Any]:
    if filepath.exists():
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def log_retrieval(
    config: dict,
    session_id: str,
    query: str,
    docs: List[Document],
) -> None:
    """
    Log metadata about retrieved documents for a query.
    """
    if not config["logging"].get("log_retrieval_metadata", False):
        return

    retrieval_dir = Path(config["paths"]["retrieval_log_dir"])
    retrieval_dir.mkdir(exist_ok=True)

    filepath = retrieval_dir / f"{session_id}_retrieval.json"

    data = _safe_load_json(filepath)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "model": config["models"]["chat_model"],       
        "question": query,
        "documents": [
            {
                "title": d.metadata.get("title"),
                "source": d.metadata.get("source"),
                "section": d.metadata.get("section"),
                "page": d.metadata.get("page"),
            }
             
            for d in docs
        ],
    }
    
    data.append(entry)


    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

# conversation Logger
        
def save_conversation(
    config: dict,
    session_id: str,
    question: str,
    answer: str,
) -> None:
    if not config["logging"]["log_conversations"]:
        return

    convo_dir = Path(config["paths"]["conversation_dir"])
    convo_dir.mkdir(exist_ok=True)

    filepath = convo_dir / f"{session_id}.json"

    if filepath.exists():
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append({
        "timestamp": datetime.now().isoformat(),
        "model": config["models"]["chat_model"],
        "question": question,
        "answer": answer,
    })

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


