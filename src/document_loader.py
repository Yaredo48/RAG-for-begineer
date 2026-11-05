from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_documents(data_dir="data/raw"):

    documents = []

    data_path = Path(data_dir)

    for file_path in data_path.rglob("*"):

        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        try:
            if extension == ".pdf":
                loader = PyPDFLoader(str(file_path))

            elif extension == ".docx":
                loader = Docx2txtLoader(str(file_path))

            else:
                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8",
                    autodetect_encoding=True
                )

            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata.update({
                    "source": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "file_type": extension
                })

            documents.extend(loaded_docs)

            print(f"Loaded: {file_path.name}")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return documents