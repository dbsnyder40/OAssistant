# oassistant/chat/session.py

from oassistant.chat.logging import save_conversation
from typing import Any

# =============================
# Step 5: Interactive session
# =============================

def interactive_session(rag_chain: Any, config: dict) -> None:

    print("\nResearch Assistant Ready!")
    print("Type 'exit' or 'quit' to stop.\n")

    default_session = config["session"]["default_session_id"]
    session_prompt = f"Session name [{default_session}]: "
    session_id = input(session_prompt).strip() or default_session
    
    while True:
        query = input("Ask a question about the documents: ").strip()
        if query.lower() in ("exit", "quit"):
            break

        response = rag_chain.invoke(
            {"question": query},
            config={"configurable": {"session_id": session_id}}
        )

        print("\n", response.content, "\n")

        save_conversation(
            config,
            session_id,
            query,
            response.content
        )
  


