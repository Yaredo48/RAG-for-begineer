import hashlib

from langchain_chroma import Chroma


def create_vector_store(chunks, embeddings):

    vector_db = Chroma(
        collection_name="rag_documents",
        embedding_function=embeddings,
        persist_directory="./vector_db"
    )

    ids = []

    for chunk in chunks:

        unique_text = (
            chunk.metadata.get("source", "")
            + str(chunk.metadata.get("page", ""))
            + str(chunk.metadata.get("start_index", ""))
            + chunk.page_content
        )

        chunk_id = hashlib.sha256(
            unique_text.encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)

    vector_db.add_documents(
        documents=chunks,
        ids=ids
    )

    print("Documents successfully stored in ChromaDB!")

    return vector_db