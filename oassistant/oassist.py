from oassistant.config import load_config
from oassistant.indexing import get_or_create_vectorstore
from oassistant.chat import create_rag_chain, interactive_session
from oassistant.version import VERSION

def main():
    args = parse_args()
    config = load_config(args.config)
    apply_cli_overrides(config, args)

    vectorstore, chunks = get_or_create_vectorstore(config)
    rag_chain = create_rag_chain(vectorstore, chunks, config)

    interactive_session(rag_chain, config)

if __name__ == "__main__":
    main()
