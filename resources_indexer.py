# resources_indexer.py
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
# FIX: Correct import for Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter 

# Load environment variables from .env file
load_dotenv()

# CRITICAL FIX 1: Prioritize keys, ensuring GOOGLE_API_KEY is set for embeddings
if os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
elif os.getenv("GOOGLE_API_KEY"):
    GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
else:
    GEMINI_KEY = None


def index_resources(index_name="learning-materials"):
    """Loads documents, splits them, and indexes them in ChromaDB."""
    
    api_key_status = 'Loaded' if GEMINI_KEY else 'MISSING'
    print(f"DEBUG: Attempting to initialize embeddings. API Key Status: {api_key_status}")
    
    if api_key_status == 'MISSING':
        print("ERROR: GEMINI_API_KEY is missing. Check your .env file.")
        return None
        
    try:
        # CRITICAL FIX 2: Explicitly pass the API key to bypass ADC lookup
        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004", 
            api_key=GEMINI_KEY  
        ) 
        print("DEBUG: Embeddings model initialized successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Authentication failed. Full Error: {e}")
        print("ACTION: Your API key is being rejected by the service. Please verify the key is correctly copied into .env.")
        return None

    # Load Documents from a "docs/" directory
    try:
        loader = DirectoryLoader('./docs', glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        if not documents:
            print("WARNING: No documents loaded from the ./docs directory. Indexing skipped.")
            return None
    except Exception as e:
        print(f"ERROR: Could not load documents from ./docs. Error: {e}")
        return None

    # Split documents for embedding
    # FIX: Corrected the typo in the class name
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    
    # ChromaDB Setup (Local Persistence)
    chroma_path = os.getenv('CHROMA_DB_PATH', './chroma_data')
    
    # Create the vector store
    print(f"Indexing {len(texts)} chunks...")
    vectorstore = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=chroma_path,
        collection_name='learning-materials'
    )
    vectorstore.persist()
    print(f"✅ SUCCESSFULLY INDEXED {len(texts)} chunks into ChromaDB at {chroma_path}")
    return vectorstore


def get_resource_retriever(vectorstore):
    """Returns a retriever instance for the LangChain Agent to use."""
    if vectorstore is None:
        raise ValueError("Vector store is not initialized.")
    return vectorstore.as_retriever(search_kwargs={"k": 3})


if __name__ == '__main__':
    # Test authentication and indexing in isolation
    index_resources()