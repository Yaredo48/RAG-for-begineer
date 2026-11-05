import re

from langchain_core.documents import Document


def clean_documents(documents):

    cleaned_documents = []

    for doc in documents:

        text = doc.page_content

        # Normalize line endings
        text = text.replace("\r\n", "\n")

        # Remove repeated spaces and tabs
        text = re.sub(r"[ \t]+", " ", text)

         # Remove excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

         # Remove leading and trailing spaces
        text = text.strip()

        if not text:
            continue


        cleaned_documents.append(
            documents(
                page_content=text,
                metadata=doc.metadata
            )
        )

    return cleaned_documents