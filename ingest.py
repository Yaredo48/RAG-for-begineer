from src.document_loader import load_documents
from src.text_cleaner import clean_documents
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store


def main():

    print("\n===== RAG DATA INGESTION PIPELINE =====\n")

    # 1. Load documents
    print("[1] Loading documents...")

    documents = load_documents("data/raw")

    if not documents:
        print("No documents found. Add files to data/raw/")
        return

    print(f"Loaded {len(documents)} document pages/records")


    # 2. Clean documents
    print("\n[2] Cleaning documents...")

    cleaned_docs = clean_documents(documents)

    print(f"Cleaned documents: {len(cleaned_docs)}")


    # 3. Split documents
    print("\n[3] Splitting documents into chunks...")

    chunks = split_documents(cleaned_docs)


    # 4. Generate embedding model
    print("\n[4] Initializing embedding model...")

    embeddings = get_embedding_model()

    # 5. Store in vector database
    print("\n[5] Storing embeddings in ChromaDB...")

    vector_db = create_vector_store(chunks, embeddings)


    print("\n===== INGESTION COMPLETED =====")

    print(f"Total chunks stored: {len(chunks)}")

    print(f"Vector database: {vector_db}")


if __name__ == "__main__":
    main()