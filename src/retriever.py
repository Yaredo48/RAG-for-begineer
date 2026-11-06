from langchain_chroma import Chroma

from src.embeddings import get_embedding_model


def initialize_retriever():

    # Load the embedding model
    embeddings = get_embedding_model()

    # Connect to existing ChromaDB
    vector_db = Chroma(
        collection_name="rag_documents",
        embedding_function=embeddings,
        persist_directory="./vector_db"
    )

    return vector_db


def retrieve_documents(query, k=3):

    # Initialize vector database
    vector_db = initialize_retriever()

    # Search for relevant document chunks
    results = vector_db.similarity_search_with_score(
        query,
        k=k
    )

    return results


def display_results(results):

    for i, (document, distance) in enumerate(results):

        print(f"\n========== RESULT {i + 1} ==========")

        print("\nDocument Content:")
        print(document.page_content)

        print("\nSource:")
        print(document.metadata.get("source"))

        print("\nPage:")
        print(document.metadata.get("page"))

        print("\nVector Distance:")
        print(distance)


if __name__ == "__main__":

    question = input("\nEnter your question: ")

    retrieved_results = retrieve_documents(
        query=question,
        k=3
    )

    display_results(retrieved_results)