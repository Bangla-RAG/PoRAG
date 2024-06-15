import argparse
import logging
from porag import BanglaRAGChain
import warnings

warnings.filterwarnings("ignore")
# Default constants for the script
DEFAULT_CHAT_MODEL_ID = "hassanaliemon/bn_rag_llama3-8b"
DEFAULT_EMBED_MODEL_ID = "l3cube-pune/bengali-sentence-similarity-sbert"
DEFAULT_K = 4
DEFAULT_TOP_K = 2
DEFAULT_TOP_P = 0.6
DEFAULT_TEMPERATURE = 0.6
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_HF_TOKEN = None


def main():
    """
    Main function to run the Bangla Retrieval-Augmented Generation (RAG) System.
    It parses command-line arguments, loads the RAG model, and processes user queries in an interactive loop.
    """
    # Argument parser for command-line options, arguments and sub-commands
    parser = argparse.ArgumentParser(
        description="Bangla Retrieval-Augmented Generation System"
    )
    parser.add_argument(
        "--chat_model",
        type=str,
        default=DEFAULT_CHAT_MODEL_ID,
        help="The Hugging Face model ID of the chat model",
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default=DEFAULT_EMBED_MODEL_ID,
        help="The Hugging Face model ID of the embedding model",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help="The number of documents to retrieve"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="The top_k parameter for the chat model",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=DEFAULT_TOP_P,
        help="The top_p parameter for the chat model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="The temperature parameter for the chat model",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="The maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="The chunk size for text splitting",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="The chunk overlap for text splitting",
    )
    parser.add_argument(
        "--text_path",
        type=str,
        required=True,
        help="The txt file path to the text file",
    )
    parser.add_argument(
        "--show_context",
        action="store_true",
        help="Whether to show the retrieved context or not.",
    )
    parser.add_argument(
        "--quantization",
        action="store_true",
        help="Whether to enable quantization(4bit) or not.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=DEFAULT_HF_TOKEN,
        help="Your Hugging Face API token",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Initialize and load the RAG model
        rag_chain = BanglaRAGChain()
        rag_chain.load(
            chat_model_id=args.chat_model,
            embed_model_id=args.embed_model,
            text_path=args.text_path,
            k=args.k,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            hf_token=args.hf_token,
            max_new_tokens=args.max_new_tokens,
            quantization=args.quantization,
        )
        logging.info(
            f"RAG model loaded successfully: chat_model={args.chat_model}, embed_model={args.embed_model}"
        )

        # Interactive loop for user queries
        while True:
            query = input("আপনার প্রশ্ন: ")
            if query.lower() in ["exit", "quit"]:
                print("আবার দেখা হবে, ধন্যবাদ!")
                break
            try:
                answer, context = rag_chain.get_response(query)
                if args.show_context:
                    print(f"প্রসঙ্গঃ {context}\n------------------------\n")
                print(f"উত্তর: {answer}")
            except Exception as e:
                logging.error(f"Couldn't generate an answer: {e}")
                print("আবার চেষ্টা করুন!")

    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        print("Error occurred, please check logs for details.")


# Entry point for the script
if __name__ == "__main__":
    main()
