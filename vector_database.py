from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

# ================================
# Step 1: Load PDF
# ================================

PDF_PATH = "universal_declaration_of_human_rights.pdf"
FAISS_DB_PATH = "vectorstore/db_faiss"

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

documents = load_pdf(PDF_PATH)
print(f"[INFO] Pages loaded: {len(documents)}")


# ================================
# Step 2: Create Chunks
# ================================

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return splitter.split_documents(documents)

text_chunks = create_chunks(documents)
print(f"[INFO] Chunks created: {len(text_chunks)}")


# ================================
# Step 3: Embeddings (ONLY THIS MODEL)
# ================================

EMBEDDING_MODEL = "nomic-embed-text"

def get_embedding_model():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


# ================================
# Step 4: FAISS Vector Store
# ================================

os.makedirs("vectorstore", exist_ok=True)

faiss_db = FAISS.from_documents(
    text_chunks,
    get_embedding_model()
)

faiss_db.save_local(FAISS_DB_PATH)

print("[SUCCESS] FAISS vector database created successfully 🎉")
