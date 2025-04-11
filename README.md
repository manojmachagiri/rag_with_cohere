# PDF RAG Application with Cohere

A Streamlit-based application that allows users to upload PDF documents, process them, and ask questions using Retrieval-Augmented Generation (RAG) with Cohere's powerful language models.

## Features

- PDF document upload and text extraction
- Text chunking and vector embedding using Cohere's embedding models
- Vector storage with ChromaDB
- Question answering using RAG with Cohere's LLMs
- Simple setup with Cohere API key

## Architecture

This application follows a modular architecture:

```
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── utils/                  # Utility modules
│   ├── __init__.py         # Package initialization
│   ├── pdf_processor.py    # PDF processing utilities
│   ├── vector_store.py     # Vector database operations
│   ├── cohere_integration.py # Cohere API integration
│   └── cohere_embeddings.py # Cohere embeddings for LangChain
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   ├── test_pdf_processor.py
│   └── test_vector_store.py
└── config.py               # Configuration settings
```

## Prerequisites

- Python 3.9+
- Cohere API key (sign up at [cohere.com](https://cohere.com/))

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pdf-rag-cohere.git
   cd pdf-rag-cohere
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Update the Cohere API key in `config.py`:
   ```python
   # Open config.py and replace the API key with your own
   COHERE_API_KEY = "your-api-key-here"
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. In the sidebar, click "Connect to Cohere" to establish a connection with the Cohere API

4. Select a model from the dropdown menu (if you want to change from the default model)

5. Upload a PDF file and click "Process PDF" to extract and index the content

6. Ask questions about the PDF content in the text input field and click "Get Answer"

## Configuration

You can modify the application's behavior by adjusting the settings in the sidebar:

- **Chunk Size**: Controls the size of text chunks (default: 1000 characters)
- **Chunk Overlap**: Controls the overlap between chunks (default: 200 characters)
- **Number of Results**: Controls how many chunks are retrieved for context (default: 4)

## How It Works

1. **PDF Processing**: The application extracts text from uploaded PDFs using PyPDF2
2. **Text Chunking**: The extracted text is split into manageable chunks with specified overlap
3. **Embedding Generation**: Each chunk is converted into a vector embedding using Cohere's embedding models
4. **Vector Storage**: Embeddings are stored in a ChromaDB vector database
5. **Query Processing**: When a question is asked, it's converted to an embedding and similar chunks are retrieved
6. **Answer Generation**: The retrieved chunks and the question are sent to Cohere for answer generation

## Troubleshooting

- **API Key Issues**: Ensure your Cohere API key is correctly set in config.py
- **Connection Issues**: Check your internet connection as the app requires access to Cohere's API
- **Memory Issues**: Reduce chunk size or number of results if you encounter memory problems
- **Rate Limits**: Be aware of Cohere API rate limits if processing many documents or queries

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
