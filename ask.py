from src.retriever import initialize_retriever
from src.generator import generate_answer


def main():

    print("\n===== RAG QUESTION ANSWERING SYSTEM =====")

    vector_db = initialize_retriever()

    while True:

        question = input("\nAsk your question (or type exit): ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question.strip():
            continue

        # STEP 1: Retrieve relevant documents

        print("\nRetrieving relevant documents...")

        documents = vector_db.similarity_search(
            question,
            k=3
        )

        if not documents:
            print("No relevant documents found.")
            continue

        # STEP 2: Generate answer

        print("\nGenerating answer...")

        answer = generate_answer(
            question,
            documents
        )

        # STEP 3: Display answer

        print("\n========== ANSWER ==========\n")

        print(answer)

        # STEP 4: Display sources

        print("\n========== SOURCES ==========\n")

        for i, doc in enumerate(documents, start=1):

            print(f"Source {i}:")
            print("File:", doc.metadata.get("file_name"))
            print("Page:", doc.metadata.get("page"))
            print()


if __name__ == "__main__":
    main()