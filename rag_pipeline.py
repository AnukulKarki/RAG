from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

DATABASE_DIR = "./rag_pipe_database"

#this creates a embedding and stores even if there is something already exists
def create_kb():
    """Create a vector store from knowledge base."""

    #split the knowledge base into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50) #creating a object of Recirsive character splitter with the information
    doc = Document(page_content=KNOWLEDGE_BASE, metadata = {
        "source": "langchain_knowledge_base.md"
    } )
    #chinking the actual documents now
    chunks = splitter.split_documents([doc])
    # print()
    # print("--"*50)
    # print(chunks)
    # print("--"*50)
    # print
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=tempfile.mkdtemp()

    )

    return vector_store

#creating a vector store that reuse in case if the vector database is already there so that it does not embedd
def get_vector_store():
    #checking if the vector store already exists if there is one then simply reuse it
    if os.path.exists(DATABASE_DIR) and os.listdir(DATABASE_DIR):
        return Chroma(
            persist_directory=DATABASE_DIR,
            embedding_function=embedding_model
        )
    #CREATING a new vector store if the old one does not exist.
    #split the knowledge base into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50) #creating a object of Recirsive character splitter with the information
    doc = Document(page_content=KNOWLEDGE_BASE, metadata = {
        "source": "langchain_knowledge_base.md"
    } )
    #chinking the actual documents now
    chunks = splitter.split_documents([doc])
        # print()
        # print("--"*50)
        # print(chunks)
        # print("--"*50)
        # print
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DATABASE_DIR   
    )
    
    return vector_store


def demo_basic_rag():
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k":2})
    print("Retriever")
    print(retriever)
    #init_chat_model can be used to use all the model in one function just by passing the mdoel name instead of importing function from each model individually
    llm = init_chat_model("groq:openai/gpt-oss-20b", temperature=0.2)

    #Rag prompt template
    prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the following context:
    {context}

    Question:{question}

    Answer:

    Make sure to answer in concise manner, and if you dont know the answer, Just say "I don't know."

    """
    )


    #Format retrieved docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    #Rag Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


    #Test 
    questions = [
        "What is Langchain?",
        "Who created Langchain?",
        "What is langgraph used for",
        "What is my Name"
    ]


    print("Basic Rag Demo")

    for q in questions:
        # Returns a list of tuples: [(Document, score), (Document, score)]
        results_with_scores = vector_store.similarity_search_with_score(q, k=2)

        print(f"Q: {q}")
        print("--- Context & Scores from DB ---")
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"Chunk {i} (Score: {score:.4f}):\n{doc.page_content}\n")
        print("--------------------------------")

        answer = rag_chain.invoke(q)
        print(f"Ans: {answer}\n")



if __name__ == "__main__":
    demo_basic_rag()