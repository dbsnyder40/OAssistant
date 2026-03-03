# oassitant/chat/rag_chain.py

"""
develop the rag_chain prompt

	format_docs
	build_prompt
	hybrid_retriever
	create_rag_chain
"""
	

# from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# from oassistant.models import create_llm
from oassistant.chat.logging import log_retrieval
from oassistant.models import create_chat_model
from typing import List

# ----------------------------
# Document formatting
# ----------------------------

def format_docs(docs: List[Document]) -> str:
    formatted = []
    for d in docs:
        meta = d.metadata

        citation_parts = []

        if meta.get("title"):
            citation_parts.append(meta["title"])

        if meta.get("source"):
            citation_parts.append(meta["source"])

        if meta.get("section"):
            citation_parts.append(f"section: {meta['section']}")

        if meta.get("page"):
            citation_parts.append(f"page: {meta['page']}")

        citation = "  ".join(citation_parts)

        formatted.append(f"""
[{citation}]
{d.page_content}
"""     )

    return "\n\n".join(formatted).strip()



# ----------------------------
# Prompt
# ----------------------------

def build_prompt() -> ChatPromptTemplate:
    """
    Returns the core RAG prompt template.
    Expects variables:
        - history
        - context
        - question
    """

    return ChatPromptTemplate.from_template("""
You are a research assistant.

Answer the question using ONLY the provided context.
If the answer is not in the context, say you do not know.

CITATION RULES:
- Every factual claim must be cited.
- Use parentheses for citations.
- Include: title, source filename, section (if available), and page (if available).
- Format citation like:
  (Title  source, section: X, page: Y)
- Omit section or page if not available.

At the end of your answer, include a section:

Sources:
- One bullet per unique document cited.

Conversation History:
{history}

Context:
{context}

Question:
{question}
"""
    )

# ----------------------------
# Hybrid Retriever
# ----------------------------
def build_hybrid_retriever(vectorstore: FAISS, chunks: List[Document], config: dict) -> RunnableLambda:
    """
    Returns a Runnable that:
    - Runs BM25 + vector retrieval
    - Deduplicates results
    - Logs retrieval metadata
    - Returns list[Document]
    """

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": config["retrieval"]["vector_k"]}
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = config["retrieval"]["bm25_k"]

    def hybrid_retriever_fn(x):
        query = x["question"]
        session_id = x.get("session_id", "default")

        bm25_docs = bm25_retriever.invoke(query)
        vector_docs = vector_retriever.invoke(query)

        # Merge results
        all_docs = bm25_docs + vector_docs

        # Deduplicate by content
        unique_docs = {doc.page_content: doc for doc in all_docs}.values()
        docs = list(unique_docs)

        # Log retrieval
        log_retrieval(config, session_id, query, docs)

        return docs

    return RunnableLambda(hybrid_retriever_fn)

# ----------------------------
# RAG Chain
# ----------------------------

def create_rag_chain(vectorstore, chunks, config) -> RunnableWithMessageHistory:

    #llm = ChatOllama(
    #    model=config["models"]["chat_model"],
    #    temperature=config["models"]["temperature"]
    #)
    llm = create_chat_model(config)
    # rag_chain = create_rag_chain(vectorstore, chunks, llm, config)

#    def retrieval_with_logging(x):
#        query = x["question"]
#        session_id = x.get("session_id", "default")
#
#        docs = hybrid_retriever_fn(query)
#        log_retrieval(config, session_id, query, docs)
#        return docs

    hybrid_runnable = build_hybrid_retriever(vectorstore, chunks, config)
    formatter_runnable = RunnableLambda(format_docs)

#------
    prompt = build_prompt()

#--------
    rag_chain = (
        {
            "context": hybrid_runnable | formatter_runnable,
            "question": RunnableLambda(lambda x: x["question"]),
            "history": RunnableLambda(lambda x: x.get("history", "")),
        }
        | prompt
        | llm
    )

    store = {}

    def get_session_history(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()       
        return store[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    ) 
