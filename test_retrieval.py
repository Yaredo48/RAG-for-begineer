from src.embeddings import get_embedding_model
from langchain_chroma import Chroma


# Load the existing embedding model
embeddings = get_embedding_model()

# Connect to existing ChromaDB
vector_db = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory="./vector_db"
)

# Ask a test question
query = "What is High Performance Computing?"

# Retrieve relevant document chunks
results = vector_db.similarity_search(
    query,
    k=3
)

# Display results
for i, doc in enumerate(results):

    print(f"\n===== RESULT {i+1} =====")

    print("Content:")
    print(doc.page_content)

    print("\nSource:")
    print(doc.metadata.get("source"))

    print("\nPage:")
    print(doc.metadata.get("page"))