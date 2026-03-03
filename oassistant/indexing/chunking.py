# oassistant/indexing/chunking.py

"""
    chunk documents
"""
from typing import List
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter


# =============================
# Step 2: Chunk documents
# =============================

def chunk_documents(docs: List[Document], config: dict) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
    )

    chunks = splitter.split_documents(docs)

    # Ensure metadata persists
    for chunk in chunks:
        chunk.metadata.setdefault("section", None)
        chunk.metadata.setdefault("page", None)

    return chunks

