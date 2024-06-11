# Bangla Retrieval-Augmented Generation (RAG) Pipeline
![Banner](/banner.png)

[![LinkedIn: Abdullah Al Asif](https://img.shields.io/badge/LinkedIn-Abdullah%20Al%20Asif-blue)](https://www.linkedin.com/in/abdullahalasif-bd/)
[![LinkedIn: Hasan Ali Emon](https://img.shields.io/badge/LinkedIn-Hasan%20Ali%20Emon-blue)](https://www.linkedin.com/in/hassan-ali-emon/)

Welcome to the **Bangla Retrieval-Augmented Generation (RAG) Pipeline**! This repository provides a pipeline for interacting with Bengali text data using natural language.

## Use Cases

- Interact with your Bengali data in Bengali.
- Ask questions about your Bengali text and get answers.

## How It Works

- **LLM Framework:** [Transformers](https://huggingface.co/docs/transformers/index)
- **RAG Framework:** [Langchain](https://www.langchain.com/)
- **Chunking:** [Recursive Character Split](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/)
- **Vector Store:** [ChromaDB](https://www.trychroma.com/)
- **Data Ingestion:** Currently supports text (.txt) files only due to the lack of reliable Bengali PDF parsing tools.

## Configurability

- **Customizable LLM Integration:** Supports Hugging Face or local LLMs compatible with Transformers.
- **Flexible Embedding:** Supports embedding models compatible with Sentence Transformers (embedding dimension: 768).
- **Fine-Grained Control:** Adjust `max_new_tokens`, `top_p`, `top_k`, `temperature`, `chunk_size`, `chunk_overlap`, and `k`.
- **Toggle Quantization mode:** Pass `--quantization` argument to toggle between different types of model including LoRA and 4bit quantization.

## Installation

1. **Install Python:** Download and install Python from [python.org](https://www.python.org/).
2. **Clone the Repository:**
    ```bash
    git clone https://github.com/Bangla-RAG/Bangla-RAG-Pipeline.git
    cd bangla-rag-pipeline
    ```
3. **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

<details>
<summary>Click to view example `requirements.txt`</summary>

```txt
transformers
bitsandbytes 
peft 
accelerate 
chromadb
langchain 
langchain-community
sentence_transformers
argparse
rich
```
</details>

## Running the Pipeline

1. **Prepare Your Bangla Text Corpus:** Create a text file (e.g., `text.txt`) with the Bengali text you want to use.
2. **Run the RAG Pipeline:**
    ```bash
    python main.py --text_path text.txt
    ```
3. **Interact with the System:** Type your question and press Enter to get a response based on the retrieved information.

## Example

```bash
আপনার প্রশ্ন: রবীন্দ্রনাথ ঠাকুরের জন্মস্থান কোথায়?
উত্তর: রবীন্দ্রনাথ ঠাকুরের জন্মস্থান কলকাতার জোড়াসাঁকোর 'ঠাকুরবাড়ি'তে।
```

## Configuration (Default)

- **Default Chat Model:** `hassanaliemon/bn_rag_llama3-8b`
- **Default Embedding Model:** `l3cube-pune/bengali-sentence-similarity-sbert`
- **Default `k`:** `4` (number of documents to retrieve)
- **Default `top_k`:** `2` (for chat model)
- **Default `top_p`:** `0.6` (for chat model)
- **Default `temperature`:** `0.6` (for chat model)
- **Default `chunk_size`:** `500` (for text splitting)
- **Default `chunk_overlap`:** `150` (for text splitting)
- **Default `max_new_tokens`:** `1024` (maximum length of the response messages)
- **Default `quantization`:** `False` (sets `load_in_4bit` boolean)

You can change these values in the `main.py` script.

## Key Milestones

- **Default LLM:** Trained a LLaMA-3 8B model `hassanaliemon/bn_rag_llama3-8b` for context-based QA.
- **Embedding Model:** Tested `sagorsarker/bangla-bert-base`, `csebuetnlp/banglabert`, and found `l3cube-pune/bengali-sentence-similarity-sbert` to be most effective.
- **Retrieval Pipeline:** Implemented Langchain Retrieval pipeline and tested with our fine-tuned LLM and embedding model.
- **Ingestion System:** Settled on text files after testing several PDF parsing solutions.
- **Question Answering Chat Loop:** Developed a multi-turn chat system for terminal testing.
- **Generation Configuration Control:** Attempted to use generation config in the LLM pipeline.
- **Model Testing:** Tested with the following models(quantized and lora versions):
  1. [`asif00/bangla-llama`](https://huggingface.co/asif00/bangla-llama)
  2. [`hassanaliemon/bn_rag_llama3-8b`](https://huggingface.co/hassanaliemon/bn_rag_llama3-8b)
  3. [`asif00/mistral-bangla`](https://huggingface.co/asif00/mistral-bangla)
  4. [`KillerShoaib/llama-3-8b-bangla-4bit`](https://huggingface.co/KillerShoaib/llama-3-8b-bangla-4bit)

## Limitations

- **PDF Parsing:** Currently, only text (.txt) files are supported due to the lack of reliable Bengali PDF parsing tools.
- **Model Performance:** The performance depends heavily on the quality of your chosen LLM, embedding model and your Bengali text corpus.

## Future Steps

- **PDF Parsing:** Develop a reliable Bengali-specific PDF parser.
- **User Interface:** Design a chat-like UI for easier interaction.
- **Chat History Management:** Implement a system for maintaining and accessing chat history.

## Contribution and Feedback

We welcome contributions! If you have suggestions, bug reports, or enhancements, please open an issue or submit a pull request.

### Contributors
[![LinkedIn: Abdullah Al Asif](https://img.shields.io/badge/LinkedIn-Abdullah%20Al%20Asif-blue)](https://www.linkedin.com/in/abdullahalasif-bd/) [Abdullah Al Asif](https://github.com/himisir)

[![LinkedIn: Hasan Ali Emon](https://img.shields.io/badge/LinkedIn-Hasan%20Ali%20Emon-blue)](https://www.linkedin.com/in/hassan-ali-emon/) [Hasan Ali Emon](https://github.com/hassanaliemon)



## Disclaimer

This is a work-in-progress and may require further refinement. The results depend on the quality of your Bengali text corpus and the chosen models.


### References

1. [Transformers](https://huggingface.co/docs/transformers/index)
2. [Langchain](https://www.langchain.com/)
3. [ChromaDB](https://www.trychroma.com/)
4. [Sentence Transformers](https://www.sbert.net/)
5. [hassanaliemon/bn_rag_llama3-8b](https://huggingface.co/hassanaliemon/bn_rag_llama3-8b)
6. [l3cube-pune/bengali-sentence-similarity-sbert](https://huggingface.co/l3cube-pune/bengali-sentence-similarity-sbert)
7. [sagorsarker/bangla-bert-base](https://huggingface.co/sagorsarker/bangla-bert-base)
8. [csebuetnlp/banglabert](https://huggingface.co/csebuetnlp/banglabert)
9. [asif00/bangla-llama](https://huggingface.co/asif00/bangla-llama)
10. [KillerShoaib/llama-3-8b-bangla-4bit](https://huggingface.co/KillerShoaib/llama-3-8b-bangla-4bit)
11. [asif00/mistral-bangla](https://huggingface.co/asif00/mistral-bangla)