#!/usr/bin/env python3
from oassistant.config import load_config, parse_args
from oassistant.indexing.manifest import needs_rebuild
from oassistant.indexing.loaders import load_documents
from oassistant.indexing.chunking import chunk_documents
from oassistant.indexing.vectorstore import build_vectorstore
from oassistant.models import create_embeddings, create_chat_model
from oassistant.chat.rag_chain import create_rag_chain
from oassistant.chat.session import interactive_session

def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.model:
        config["models"]["chat_model"] = args.model
    if args.documents:
        config["paths"]["papers_dir"] = args.documents

    # Load/build index
    if needs_rebuild(config):
        documents = load_documents(config["paths"]["papers_dir"], config["corpus"])
        chunks = chunk_documents(documents, config)
        vectorstore = build_vectorstore(chunks, config)
    else:
        embeddings = create_embeddings(config)
        vectorstore = FAISS.load_local(config["paths"]["vectorstore_dir"], embeddings, allow_dangerous_deserialization=True)
        documents = load_documents(config["paths"]["papers_dir"], config["corpus"])
        chunks = chunk_documents(documents, config)

    rag_chain = create_rag_chain(vectorstore, chunks, config)
    interactive_session(rag_chain, config)

if __name__ == "__main__":
    main()
