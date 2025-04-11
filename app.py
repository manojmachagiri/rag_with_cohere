"""Main Streamlit application for PDF RAG with Cohere.

This application allows users to:
1. Upload PDF documents
2. Process and extract text
3. Create vector embeddings using Cohere
4. Ask questions about the PDF content
5. Get answers using RAG with Cohere's LLMs
"""

import streamlit as st # type: ignore
import traceback

from utils.pdf_processor import save_uploaded_file, extract_text_from_pdf, split_text
from utils.vector_store import create_vector_store, clear_vector_store, similarity_search
from utils.cohere_embeddings import CohereEmbeddings
from utils.cohere_integration import CohereClient
from logger import setup_logger
from config import (
    APP_TITLE, APP_DESCRIPTION, APP_ICON, APP_LAYOUT, SIDEBAR_STATE,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_NUM_RESULTS, SYSTEM_PROMPT,
    COHERE_EMBEDDING_MODEL, COHERE_GENERATION_MODEL
)

# Set up logger
logger = setup_logger(__name__)

# Set page configuration from config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state=SIDEBAR_STATE
)

# Initialize session state variables
if "processed_file" not in st.session_state:
    st.session_state.processed_file = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "cohere_client" not in st.session_state:
    st.session_state.cohere_client = None
if "available_models" not in st.session_state:
    st.session_state.available_models = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# Function to initialize Cohere client and get available models
def initialize_cohere():
    """Initialize the Cohere client and get available models.

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Initializing Cohere client")
    try:
        cohere_client = CohereClient()
        models = cohere_client.list_models()
        model_names = [model.get("name") for model in models]

        logger.info(f"Found {len(model_names)} available models")
        st.session_state.available_models = model_names

        if model_names:
            st.session_state.selected_model = model_names[0]
            cohere_client.generation_model = model_names[0]
            logger.info(f"Selected model: {model_names[0]}")
        else:
            logger.warning("No models available in Cohere")

        st.session_state.cohere_client = cohere_client
        return True
    except Exception as e:
        logger.error(f"Error initializing Cohere: {e}")
        st.error(f"Error initializing Cohere: {e}")
        st.info("Make sure your Cohere API key is correct")
        return False

# Main title from config
st.title(APP_TITLE)
st.markdown(APP_DESCRIPTION)

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # Initialize Cohere
    if st.button("Connect to Cohere"):
        with st.spinner("Connecting to Cohere..."):
            if initialize_cohere():
                st.success("Connected to Cohere successfully!")

    # Model selection if models are available
    if st.session_state.available_models:
        selected_model = st.selectbox(
            "Select Cohere Model",
            options=st.session_state.available_models,
            index=st.session_state.available_models.index(st.session_state.selected_model)
        )

        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            if st.session_state.cohere_client:
                st.session_state.cohere_client.generation_model = selected_model
                st.success(f"Model changed to {selected_model}")

    # Advanced settings with defaults from config
    st.subheader("Advanced Settings")
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=DEFAULT_CHUNK_SIZE, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=DEFAULT_CHUNK_OVERLAP, step=50)
    num_results = st.slider("Number of Results to Retrieve", min_value=1, max_value=10, value=DEFAULT_NUM_RESULTS)

    # Add a button to clear the vector store
    if st.button("Clear Vector Store"):
        with st.spinner("Clearing vector store..."):
            if clear_vector_store():
                st.session_state.vector_store = None
                st.session_state.processed_file = False
                st.success("Vector store cleared successfully!")
                logger.info("Vector store cleared by user")
            else:
                st.error("Error clearing vector store")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Process PDF"):
            try:
                with st.spinner("Processing PDF..."):
                    logger.info(f"Processing PDF: {uploaded_file.name}")

                    # Save the uploaded file
                    temp_file_path = save_uploaded_file(uploaded_file)
                    logger.debug(f"Saved uploaded file to: {temp_file_path}")

                    # Extract text from PDF
                    text = extract_text_from_pdf(temp_file_path)
                    st.session_state.extracted_text = text
                    logger.info(f"Extracted {len(text)} characters from PDF")

                    # Split text into chunks
                    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.session_state.chunks = chunks
                    logger.info(f"Split text into {len(chunks)} chunks")

                    # Check if Cohere client is initialized
                    if not st.session_state.cohere_client:
                        st.warning("Please connect to Cohere first before processing the PDF.")
                        st.stop()

                    # Create vector store using Cohere embeddings
                    try:
                        logger.info(f"Creating vector store with model: {COHERE_EMBEDDING_MODEL}")

                        # Create embeddings model using Cohere
                        embeddings = CohereEmbeddings()

                        # Create vector store (this will clear any existing store)
                        vector_store = create_vector_store(chunks, embeddings=embeddings)
                        st.session_state.vector_store = vector_store
                        st.session_state.processed_file = True

                        logger.info("PDF processing completed successfully")
                        st.success(f"PDF processed successfully! Extracted {len(chunks)} chunks.")
                    except Exception as e:
                        st.error(f"Error creating vector store: {e}")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                st.error(traceback.format_exc())

    # Display text preview if available
    if st.session_state.processed_file:
        with st.expander("Preview Extracted Text"):
            st.text_area("Extracted Text", st.session_state.extracted_text[:1000] + "...", height=300)

with col2:
    st.header("Ask Questions")

    # Check if PDF is processed and Cohere is connected
    if not st.session_state.processed_file:
        st.info("Please upload and process a PDF first.")
    elif not st.session_state.cohere_client:
        st.info("Please connect to Cohere first.")
    else:
        # Question input
        query = st.text_input("Ask a question about the PDF content")

        if query and st.button("Get Answer"):
            try:
                with st.spinner("Searching for relevant information..."):
                    # Search for relevant documents
                    docs = similarity_search(st.session_state.vector_store, query, k=num_results)

                    # Display retrieved context
                    with st.expander("Retrieved Context", expanded=False):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.text(doc.page_content)
                            st.markdown("---")

                with st.spinner(f"Generating answer with Cohere {st.session_state.selected_model}..."):
                    logger.info(f"Generating answer for query: {query}")
                    # Generate answer using Cohere with system prompt from config
                    answer = st.session_state.cohere_client.generate_with_context(
                        query,
                        docs,
                        system_prompt=SYSTEM_PROMPT
                    )
                    logger.info("Answer generated successfully")

                    # Display answer
                    st.markdown("### Answer")
                    st.markdown(answer)
            except Exception as e:
                st.error(f"Error generating answer: {e}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain, and Cohere")

# Add version information
st.sidebar.markdown("---")
st.sidebar.caption("v1.0.0 | PDF RAG with Cohere")
