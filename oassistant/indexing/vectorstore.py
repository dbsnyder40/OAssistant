# oassistant/index/vectorstore.py
""" create embedding from documents """

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path


# =============================
# Step 3: Create or load FAISS
# =============================

   
def build_vectorstore(
    chunks: List[Document],
    config: dict
) -> FAISS:
    embeddings = OllamaEmbeddings(
        model=config["models"]["embedding_model"]
    )

    db_dir = Path(config["paths"]["vectorstore_dir"])

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(db_dir))
    return vectorstore
    
