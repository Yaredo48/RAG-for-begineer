from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from src.prompt_template import RAG_PROMPT


def initialize_llm():

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    return llm


def format_context(documents):

    context_parts = []

    for i, doc in enumerate(documents, start=1):

        source = doc.metadata.get("file_name", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"""
            [Source {i}]
            Document: {source}
            Page: {page}

            Content:
            {doc.page_content}
            """
        )

    return "\n\n".join(context_parts)


def generate_answer(question, documents):

    # Prepare document context
    context = format_context(documents)

    # Initialize LLM
    llm = initialize_llm()

    # Connect prompt and LLM
    chain = RAG_PROMPT | llm | StrOutputParser()

    # Generate answer
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer