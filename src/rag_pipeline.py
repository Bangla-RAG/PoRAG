import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from rich import print as rprint
from rich.panel import Panel
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class BanglaRAGChain:
    """
    Bangla Retrieval-Augmented Generation (RAG) Chain for question answering.

    This class uses a HuggingFace/local language model for text generation, a Chroma vector database for
    document retrieval, and a custom prompt template to create a RAG chain that can generate
    responses to user queries in Bengali.
    """

    def __init__(self):
        """Initializes the BanglaRAGChain with default parameters."""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chat_model_id = None
        self.embed_model_id = None
        self.k = 4
        self.max_new_tokens = 1024
        self.chunk_size = 500
        self.chunk_overlap = 150
        self.text_file_path = ""
        self.temperature = 0.9
        self.top_p = 0.6
        self.top_k = 50
        self._text_content = None
        self.hf_token = None

        self.tokenizer = None
        self.chat_model = None
        self._llm = None
        self._retriever = None
        self._db = None
        self._documents = []
        self._chain = None

    def load(
        self,
        chat_model_id,
        embed_model_id,
        text_file_path,
        k=4,
        top_k=50,
        top_p=0.6,
        max_new_tokens=1024,
        temperature=0.9,
        chunk_size=500,
        chunk_overlap=150,
        hf_token=None,
    ):
        """
        Loads the required models and data for the RAG chain.

        Args:
            chat_model_id (str): The Hugging Face model ID for the chat model.
            embed_model_id (str): The Hugging Face model ID for the embedding model.
            text_file_path (str): Path to the text file to be indexed.
            k (int): The number of documents to retrieve.
            top_k (int): The top_k parameter for the generation configuration.
            top_p (float): The top_p parameter for the generation configuration.
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float): The temperature parameter for the generation configuration.
            chunk_size (int): The chunk size for text splitting.
            chunk_overlap (int): The chunk overlap for text splitting.
            hf_token (str): The Hugging Face token for authentication.
        """
        self.chat_model_id = chat_model_id
        self.embed_model_id = embed_model_id
        self.k = k
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_file_path = text_file_path
        self.max_new_tokens = max_new_tokens
        self.hf_token = hf_token

        if self.hf_token is not None:
            os.environ["HF_TOKEN"] = str(self.hf_token)

        rprint(Panel("[bold green]Loading models...", expand=False))
        self._load_models()

        rprint(Panel("[bold green]Creating document...", expand=False))
        self._create_document()

        rprint(Panel("[bold green]Updating Chroma database...", expand=False))
        self._update_chroma_db()

        rprint(Panel("[bold green]Initializing retriever...", expand=False))
        self._get_retriever()

        rprint(Panel("[bold green]Initializing LLM...", expand=False))
        self._get_llm()
        rprint(Panel("[bold green]Creating chain...", expand=False))
        self._create_chain()

    def _load_models(self):
        """Loads the chat model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.chat_model_id)
            self.chat_model = AutoModelForCausalLM.from_pretrained(
                self.chat_model_id, device_map="auto"
            )
            rprint(Panel("[bold green]Models loaded successfully!", expand=False))
        except Exception as e:
            print(f"Error loading chat model: {e}")

    def _create_document(self):
        """Splits the input text into chunks using RecursiveCharacterTextSplitter."""
        try:
            with open(self.text_file_path, "r", encoding="utf-8") as file:
                self._text_content = file.read()
            character_splitter = RecursiveCharacterTextSplitter(
                separators=["!", "?", "।"],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            self._documents = list(
                tqdm(
                    character_splitter.split_text(self._text_content),
                    desc="Chunking text",
                )
            )
            print(f"Number of chunks: {len(self._documents)}")
            if False:
                for i, chunk in enumerate(self._documents):
                    if i > 5:
                        break
                    print(f"Chunk {i}: {chunk}")
            rprint(Panel("[bold green]Document created successfully!", expand=False))
        except Exception as e:
            print(f"Chunking failed: {e}")

    def _update_chroma_db(self):
        """Updates the Chroma vector database with the text chunks."""
        try:
            model_kwargs = {"device": self._device}
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embed_model_id, model_kwargs=model_kwargs
            )
            self._db = Chroma.from_texts(texts=self._documents, embedding=embeddings)
            rprint(
                Panel("[bold green]Chroma database updated successfully!", expand=False)
            )
        except Exception as e:
            print(f"Vector DB initialization failed: {e}")

    def _create_chain(self):
        """Creates the retrieval-augmented generation (RAG) chain."""
        template = """Below is an instruction in Bengali language that describes a task, paired with an input also in Bengali language that provides further context. Write a response in Bengali that appropriately completes the request.

        ### Instruction:
        {question}

        ### Input:
        {context}

        ### Response:
        """

        prompt_template = ChatPromptTemplate(
            input_variables=["question", "context"],
            output_parser=None,
            partial_variables={},
            messages=[
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=["question", "context"],
                        output_parser=None,
                        partial_variables={},
                        template=template,
                        template_format="f-string",
                        validate_template=True,
                    ),
                    additional_kwargs={},
                )
            ],
        )

        try:
            rag_chain_from_docs = (
                RunnablePassthrough.assign(
                    context=lambda x: self._format_docs(x["context"])
                )
                | prompt_template
                | self._llm
                | StrOutputParser()
            )

            rag_chain_with_source = RunnableParallel(
                {"context": self._retriever, "question": RunnablePassthrough()}
            ).assign(answer=rag_chain_from_docs)

            self._chain = rag_chain_with_source
            rprint(Panel("[bold green]RAG chain created successfully!", expand=False))
        except Exception as e:
            print(f"RAG chain initialization failed: {e}")

    def _get_llm(self):
        """Initializes the language model for the generation."""
        try:
            generation_kwargs = {}
            config = GenerationConfig(
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                **generation_kwargs,
            )
            pipe = pipeline(
                "text-generation",
                model=self.chat_model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                truncation=True,
                # generation_config=config, //Disabled for now, causing issues.
                device_map="auto",
            )
            self._llm = HuggingFacePipeline(pipeline=pipe)
            rprint(Panel("[bold green]LLM initialized successfully!", expand=False))
        except Exception as e:
            print(f"LLM initialization failed: {e}")

    def _get_retriever(self):
        """Initializes the retriever for document retrieval."""
        try:
            self._retriever = self._db.as_retriever(
                search_type="similarity", search_kwargs={"k": self.k}
            )
            rprint(
                Panel("[bold green]Retriever initialized successfully!", expand=False)
            )
        except Exception as e:
            print(f"Retriever initialization failed: {e}")

    def _format_docs(self, docs):
        """Formats the retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def get_response(self, query):
        """
        Generates a response to the query using the RAG chain.

        Args:
            query (str): The input query.

        Returns:
            tuple: A tuple containing the generated response and the retrieved context.
        """
        try:
            response = self._chain.invoke(query)
            response_start = response["answer"].find("### Response:") + len(
                "### Response:"
            )
            final_answer = response["answer"][response_start:].strip()
            return final_answer, response["context"]
        except Exception as e:
            print(f"Answer generation failed: {e}")
            return None, None
