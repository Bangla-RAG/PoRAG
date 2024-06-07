# Bangla Retrieval-Augmented Generation (RAG) Pipeline

[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-orange)](https://huggingface.co/transformers/)

This project presents a useful and flexible Bangla Retrieval-Augmented Generation (RAG) system, enabling you to seamlessly interact with Bengali text data using natural language.

## Key Features

- **Customizable LLM Integration:** Supports Hugging Face or local LLMs compatible with Transformers Causal LLM.
- **Customizable Embedding:** Supports embedding models compatible with Sentence Transformers (embedding dimension: 768).
- **Fine-grained Control:** Offers control over `max_new_tokens`, `top_p`, `top_k`, `temperature`, `chunk_size`, `chunk_overlap`, and `k`.
- **Advanced Retrieval:** Uses ChromaDB for efficient storage and retrieval of embedded text representations.
- **RAG Chain:** Utilizes Langchain's RAG pipeline for information retrieval and generation.
- **User-Friendly Interface:** Can be used locally from your terminal.

## Installation

1. **Install Python:** Ensure Python is installed on your system. Download from [python.org](https://www.python.org/).
2. **Clone the Repository:**
    ```bash
    git clone https://github.com/himisir/bangla-rag-system.git
    cd bangla-rag-system
    ```
3. **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

### Example of `requirements.txt`

```
transformers
bitsandbytes 
peft 
accelerate 
chromadb
langchain 
langchain-community
sentence_transformers
argparse
```

Ensure these dependencies are listed in your `requirements.txt` for smooth installation.

## Usage

1. **Prepare Your Bangla Text Corpus:**
   - Create a text file (e.g., `bangla_corpus.txt`) containing the Bangla text you want to use for retrieval.
   - Ensure the text is encoded in UTF-8.

2. **Run the Script:**
    ```bash
    python main.py --text_path bangla_corpus.txt
    ```
    - Customize parameters as needed:
    ```bash
    python main.py --text_path bangla_corpus.txt --chat_model path/to/your/chat_model --embed_model path/to/your/embedding_model --k 10 --top_k 50 --top_p 0.95 --temperature 0.7 --chunk_size 500 --chunk_overlap 100 --max_new_tokens 1024
    ```
    - Use `--help` for a list of available arguments:
    ```bash
    python main.py --help
    ```

3. **Interact with the System:**
   - The script will prompt for a query in Bengali.
   - Type your question and press Enter.
   - The system will display a generated response based on retrieved information.

## Example

```bash
আপনার প্রশ্ন: রবীন্দ্রনাথ ঠাকুরের জন্মস্থান কোথায়?
উত্তর: রবীন্দ্রনাথ ঠাকুরের জন্মস্থান কলকাতার জোড়াসাঁকোর 'ঠাকুরবাড়ি'তে।
```

## Configuration

- **Default Chat Model:** `asif00/bangla-llama-4bit`
- **Default Embedding Model:** `l3cube-pune/bengali-sentence-similarity-sbert`
- **Default `k`:** 4 (number of documents to retrieve)
- **Default `top_k`:** 2 (for chat model)
- **Default `top_p`:** 0.3 (for chat model)
- **Default `temperature`:** 0.7 (for chat model)
- **Default `chunk_size`:** 500 (for text splitting)
- **Default `chunk_overlap`:** 150 (for text splitting)
- **Default `max_new_tokens`:** 1024 (maximum length of the response messages)

You can change these default values in the `main.py` script to customize the behavior of the system.

## Document Ingestion

### Current Limitation
- Only supports text (.txt) files due to the lack of reliable Bengali PDF parsing tools.
- PDF support will be added in the future.

## Performance Considerations

- **LLM and Embedding Model Dependency:** The RAG pipeline's performance is tied to the quality of the chosen LLM and embedding model.
- **GPU Requirements:** A CUDA-compatible GPU might be necessary for optimal performance with certain LLMs.

## Next Steps

- **PDF Parsing:** Developing a reliable Bengali-specific PDF parser to expand the pipeline's capabilities.

## Contribution and Feedback
We welcome your contributions! If you have any suggestions, bug reports, or want to enhance the system, please open an issue or submit a pull request. Your feedback helps us improve and grow this project.

## License

This project is licensed under the MIT License.

## Disclaimer

This is a working progress implementation and may require further refinement for optimal performance. The results are dependent on the quality of your Bengali text corpus and the chosen models.
