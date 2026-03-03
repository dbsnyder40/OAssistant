# oassistant/models.py

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.language_models import BaseChatModel


#def create_embeddings(config):
#    return OllamaEmbeddings(
#        model=config["models"]["embedding_model"]
#    )

def create_embeddings(config: dict):
    provider = config["models"].get("provider", "ollama")

    if provider == "ollama":
        return OllamaEmbeddings(
            model=config["models"]["embedding_model"]
        )

    raise ValueError(f"Unknown embedding provider: {provider}")

def create_chat_model(config: dict) -> BaseChatModel:

    return ChatOllama(
        model=config["models"]["chat_model"],
        temperature=config["models"]["temperature"],
    )
    
    
