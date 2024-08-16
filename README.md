# PoRAG (পরাগ), Bangla Retrieval-Augmented Generation (RAG) Pipeline
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
- **Hyperparameter Control:** Adjust `max_new_tokens`, `top_p`, `top_k`, `temperature`, `chunk_size`, `chunk_overlap`, and `k`.
- **Toggle Quantization mode:** Pass `--quantization` argument to toggle between different types of model including LoRA and 4bit quantization.

## Installation

1. **Install Python:** Download and install Python from [python.org](https://www.python.org/).
2. **Clone the Repository:**
    ```bash
    git clone https://github.com/Bangla-RAG/PoRAG.git
    cd PoRAG
    ```
3. **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

<details>
<summary>Click to view example `requirements.txt`</summary>

```txt
langchain==0.2.3
langchain-community==0.2.4
langchain-core==0.2.5
chromadb==0.5.0
accelerate==0.31.0
peft==0.11.1
transformers==4.40.1
bitsandbytes==0.41.3
sentence-transformers==3.0.1
rich==13.7.1
```
</details>

## Running the Pipeline

1. **Prepare Your Bangla Text Corpus:** Create a text file (e.g., `test.txt`) with the Bengali text you want to use.
2. **Run the RAG Pipeline:**
    ```bash
    python main.py --text_path test.txt
    ```
3. **Interact with the System:** Type your question and press Enter to get a response based on the retrieved information.

## Example

```bash
আপনার প্রশ্ন: রবীন্দ্রনাথ ঠাকুরের জন্মস্থান কোথায়?
উত্তর: রবীন্দ্রনাথ ঠাকুরের জন্মস্থান কলকাতার জোড়াসাঁকোর 'ঠাকুরবাড়ি'তে।
```

# Parameters description
You can pass these arguments and adjust their values during each runs.

<table>
    <thead>
        <tr>
            <th>Flag Name</th>
            <th>Type</th>
            <th width="50%">Description</th>
            <th width="50%">Instructions</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>chat_model</code></td>
            <td>str</td>
            <td>The ID of the chat model. It can be either a Hugging Face model ID or a local path to the model.</td>
            <td>Use the model ID from the HuggingFace model card or provide the local path to the model. The default value is set to <code>"hassanaliemon/bn_rag_llama3-8b"</code>.</td>
        </tr>
        <tr>
            <td><code>embed_model</code></td>
            <td>str</td>
            <td>The ID of the embedding model. It can be either a Hugging Face model ID or a local path to the model.</td>
            <td>Use the model ID from the HuggingFace model card or provide the local path to the model. The default value is set to <code>"l3cube-pune/bengali-sentence-similarity-sbert"</code>.</td>
        </tr>
        <tr>
            <td><code>k</code></td>
            <td>int</td>
            <td>The number of documents to retrieve.</td>
            <td>The default value is set to <code>4</code>.</td>
        </tr>
        <tr>
            <td><code>top_k</code></td>
            <td>int</td>
            <td>The top_k parameter for the chat model.</td>
            <td>The default value is set to <code>2</code>.</td>
        </tr>
        <tr>
            <td><code>top_p</code></td>
            <td>float</td>
            <td>The top_p parameter for the chat model.</td>
            <td>The default value is set to <code>0.6</code>.</td>
        </tr>
        <tr>
            <td><code>temperature</code></td>
            <td>float</td>
            <td>The temperature parameter for the chat model.</td>
            <td>The default value is set to <code>0.6</code>.</td>
        </tr>
        <tr>
            <td><code>max_new_tokens</code></td>
            <td>int</td>
            <td>The maximum number of new tokens to generate.</td>
            <td>The default value is set to <code>256</code>.</td>
        </tr>
        <tr>
            <td><code>chunk_size</code></td>
            <td>int</td>
            <td>The chunk size for text splitting.</td>
            <td>The default value is set to <code>500</code>.</td>
        </tr>
        <tr>
            <td><code>chunk_overlap</code></td>
            <td>int</td>
            <td>The chunk overlap for text splitting.</td>
            <td>The default value is set to <code>150</code>.</td>
        </tr>
        <tr>
            <td><code>text_path</code></td>
            <td>str</td>
            <td>The txt file path to the text file.</td>
            <td>This is a required field. Provide the path to the text file you want to use.</td>
        </tr>
        <tr>
            <td><code>show_context</code></td>
            <td>bool</td>
            <td>Whether to show the retrieved context or not.</td>
            <td>Use <code>--show_context</code> flag to enable this feature.</td>
        </tr>
        <tr>
            <td><code>quantization</code></td>
            <td>bool</td>
            <td>Whether to enable quantization(4bit) or not.</td>
            <td>Use <code>--quantization</code> flag to enable this feature.</td>
        </tr>
        <tr>
            <td><code>hf_token</code></td>
            <td>str</td>
            <td>Your Hugging Face API token.</td>
            <td>The default value is set to <code>None</code>. Provide your Hugging Face API token if necessary.</td>
        </tr>
    </tbody>
</table>


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
- **Quality of answers:** The qualities of answer depends heavily on the quality of your chosen LLM, embedding model and your Bengali text corpus.
- **Scarcity of Pre-trained models:** As of now, we do not have a high fidelity Bengali LLM Pre-trained models available for QA tasks, which makes it difficult to achieve impressive RAG performance. Overall performance may very depending on the model we use.  


## Future Steps

- **PDF Parsing:** Develop a reliable Bengali-specific PDF parser.
- **User Interface:** Design a chat-like UI for easier interaction.
- **Chat History Management:** Implement a system for maintaining and accessing chat history.

## Contribution and Feedback

We welcome contributions! If you have suggestions, bug reports, or enhancements, please open an issue or submit a pull request.

### Top Contributors
[![LinkedIn: Abdullah Al Asif](https://img.shields.io/badge/LinkedIn-Abdullah%20Al%20Asif-blue)](https://www.linkedin.com/in/abdullahalasif-bd/) [Abdullah Al Asif](https://github.com/asiff00)

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
