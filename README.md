# 🌟 Bangla Retrieval-Augmented Generation (RAG) Pipeline 🌟

![Banner](src/images/banner.png)
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-orange)](https://huggingface.co/transformers/)


Welcome to the **Bangla Retrieval-Augmented Generation (RAG) Pipeline**! This project presents a useful and flexible Bangla Retrieval-Augmented Generation (RAG) system, enabling you to seamlessly interact with Bengali text data using natural language.

## 🌟 Key Features

- **Customizable LLM Integration:** Supports Hugging Face or local LLMs compatible with Transformers Causal LLM.
- **Flexible Embedding:** Supports embedding models compatible with Sentence Transformers (embedding dimension: 768).
- **Fine-grained Control:** Adjust `max_new_tokens`, `top_p`, `top_k`, `temperature`, `chunk_size`, `chunk_overlap`, and `k`.
- **Advanced Retrieval:** Efficient storage and retrieval using ChromaDB.
- **RAG Chain:** Utilizes Langchain's RAG pipeline for retrieval and generation.
- **User-Friendly Interface:** Easy to use from your terminal.

## 🛠️ Installation

1. **Install Python:** Ensure Python is installed on your system. Download from [python.org](https://www.python.org/).
2. **Clone the Repository:**
    ```bash
    git clone https://github.com/himisir/bangla-rag-pipeline.git
    cd bangla-rag-pipeline
    ```
3. **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

### Example `requirements.txt`

<details>
<summary>Click to expand</summary>

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
```
</details>

## 🚀 Usage

1. **Prepare Your Bangla Text Corpus:**
   - Create a text file (e.g., `bangla_corpus.txt`) with the Bangla text you want to use.
2. **Run the RAG Pipeline:**
    ```bash
    python main.py --text_file_path path/to/bangla_corpus.txt
    ```
3. **Interact with the System:**
   - Type your question and press Enter.
   - The system will display a generated response based on retrieved information.

## 📖 Example

```bash
আপনার প্রশ্ন: রবীন্দ্রনাথ ঠাকুরের জন্মস্থান কোথায়?
উত্তর: রবীন্দ্রনাথ ঠাকুরের জন্মস্থান কলকাতার জোড়াসাঁকোর 'ঠাকুরবাড়ি'তে।
```

## ⚙️ Configuration

- **Default Chat Model:** `asif00/bangla-llama-4bit`
- **Default Embedding Model:** `l3cube-pune/bengali-sentence-similarity-sbert`
- **Default `k`:** 4 (number of documents to retrieve)
- **Default `top_k`:** 2 (for chat model)
- **Default `top_p`:** 0.3 (for chat model)
- **Default `temperature`:** 0.7 (for chat model)
- **Default `chunk_size`:** 500 (for text splitting)
- **Default `chunk_overlap`:** 150 (for text splitting)
- **Default `max_new_tokens`:** 1024 (maximum length of the response messages)

You can change these default values in the `main.py` script to customize the system's behavior.

## 🗂️ Document Ingestion

### Current Limitation
- **Only Supports Text (.txt) Files:** Due to the lack of reliable Bengali PDF parsing tools.
- **Future Plans:** Adding support for PDF files.

## ⚡ Performance Considerations

- **Model Dependency:** The RAG pipeline's performance depends on the quality of the chosen LLM and embedding model.
- **GPU Requirements:** A CUDA-compatible GPU might be necessary for optimal performance with certain LLMs.

## 🚧 Next Steps

- **PDF Parsing:** Developing a reliable Bengali-specific PDF parser to enhance the pipeline's capabilities.

## 🤝 Contribution and Feedback

We welcome your contributions! If you have suggestions, bug reports, or want to enhance the system, please open an issue or submit a pull request. Your feedback helps us improve and grow this project.


## 📜 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This is a work-in-progress implementation and may require further refinement for optimal performance. The results depend on the quality of your Bengali text corpus and the chosen models.
