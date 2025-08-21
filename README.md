# AI-Powered Chatbot with RAG

This project is an intelligent chatbot that answers questions based on the content of a provided PDF document. It leverages a powerful technique called **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware answers instead of relying on pre-trained knowledge.

## Overview

The core idea of RAG is to ground a Large Language Model (LLM) on a specific set of documents. This prevents the model from hallucinating and ensures that its responses are based solely on the information present in the source material.

### How It Works

The application follows a multi-step pipeline:
1.  **Document Loading:** A PDF document is loaded and split into smaller, manageable chunks of text.
2.  **Embedding Generation:** Each text chunk is converted into a numerical representation (a vector embedding) using a sentence-transformer model.
3.  **Vector Storage:** These embeddings are stored in a vector database (like FAISS), which allows for efficient similarity searching.
4.  **Retrieval:** When a user asks a question, their query is also converted into an embedding. The vector database is then searched to find the text chunks most semantically similar to the question.
5.  **Generation:** The original user question and the retrieved text chunks (the "context") are combined into a prompt that is sent to an LLM (like OpenAI's GPT).
6.  **Response:** The LLM generates a natural language answer based on the provided context, which is then returned to the user.

## Features

-   **Contextual Q&A:** Ask questions about any PDF document and receive answers based on its content.
-   **Extensible:** Easily adaptable to use different document types, embedding models, or LLMs.
-   **Speech-to-Text:** Integrated functionality to accept user queries via voice input.

## Technology Stack

-   **Python 3.9+**
-   **LangChain:** A framework for developing applications powered by language models.
-   **OpenAI API:** For accessing powerful generative models (e.g., GPT-3.5-turbo).
-   **FAISS:** A library for efficient similarity search and clustering of dense vectors.
-   **PyMuPDF / PyPDF2:** Libraries for extracting text from PDF documents.
-   **Flask (Optional):** Can be used to wrap the logic in a simple REST API.

## Getting Started

Follow these instructions to set up and run the chatbot on your local machine.

### Prerequisites

-   Python (Version 3.9 or higher)
-   Pip (Python package installer)

### Installation and Setup

**1. Clone the repository:**
```bash
git clone https://github.com/Mahender2023/RagPdf.git
cd RagPdf
```

**2. Create a Virtual Environment:**
It is highly recommended to use a virtual environment to manage project dependencies.
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies:**
Create a `requirements.txt` file with the following content:
```txt
langchain
openai
faiss-cpu
pypdf
python-dotenv
tiktoken
```
Then, install the packages:
```bash
pip install -r requirements.txt
```

**4. Set Up Environment Variables:**
This project requires an API key from OpenAI.
-   Create a file named `.env` in the root directory of the project.
-   Add your OpenAI API key to this file:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

### Usage

1.  **Add a Document:** Place the PDF file you want to query into a designated folder (e.g., a `data/` folder in the project root).
2.  **Update the Script:** Make sure the Python script (`app.py` or similar) points to the correct path of your PDF file.
3.  **Run the application:**
    ```bash
    python app.py
    ```
4.  **Ask Questions:** Follow the prompts in the console to ask questions about your document. The chatbot will process your query and provide an answer based on the PDF's content.
