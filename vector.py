import chromadb
import tempfile
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

# Sample documents
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]
#craeting a client to access the features of chromadb
def vectorCheck():
    client = chromadb.PersistentClient("./data")

    #this is the name of the collection similar to the name of table as in sql
    collection_name = "vehicle"

    collection = client.get_or_create_collection(collection_name)

    print(f"Collection created: {collection.name}")

    #adding data to the collection

    datas = [
        {"id":"car", "text":"Car runs on the land"},
        {"id":"Plane", "text":"Plane flies in the sky"},
        {"id":"Boat", "text":"Boat travels on water"},
        {"id":"Bus", "text":"Bus is public transport on road"},
    ]

    collection.add(
        documents = [
            "Car runs on the land",
            "Plane flies in the sky",
            "Boat travels on water",
            "Bus is public transport on road",
        ],
        ids = ["car", "plane", "boat", "bus"]


    )
    #query the collection
    query_texts = ["Which vehicle run on the road?"]


    results = collection.query(
        query_texts=[query_texts[0]],
        n_results= 2
    )

    print("This is the result: ",results)

def similarity_search_with_score():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )

        query = "Explain Vecto Stores."
        results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

        print("--"*50)
        print(results_with_scores)

        print("--"*50)


        # [(Document(id='d097e238-f0f1-4ec8-acd4-2d60babcfa5a', metadata={'topic': 'database', 'source': 'vector_guide'}, page_content='Vector stores are databases optimized for storingand searching embeddings.'), 0.4393782615661621), (Document(id='44cbdd74-4450-4e9a-b1c5-d05e943dd6b3', metadata={'topic': 'database', 'source': 'pinecone_docs'}, page_content='Pinecone is a managed vector database service for production workloads.'), 0.6765804290771484), (Document(id='d70e8251-9d91-41bc-a593-79c76acb205d', metadata={'topic': 'fundamentals', 'source': 'embeddings_guide'}, page_content='Embeddings convert text into numerical vectors for semantic similarity.'), 0.7388519048690796)]


        print(f"Top 3 results with score for query '{query}")

        for i,(doc,score) in enumerate(results_with_scores):
            print(f"Result {i+1}:  {doc.page_content} (Score: {score:.4f}, source: {doc.metadata['source']})")

def metadata_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )

        query = "What databases are available"

        #without metadata filtering
        results = vectorstore.similarity_search(query, k = 5)
        print(f"Results without metadata filtering for query '{query}':")
        for i, doc in enumerate(results):
            print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

        print()
        print("--"*50)
        print()

        #with metadata filtering
        filter_criteria = {"topic":"database"}
        filter_results = vectorstore.similarity_search(query, k = 5, filter=filter_criteria)
        print(f"Results  with metadata filtering for query '{query}':")
        for i, doc in enumerate(filter_results):
            print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")



if __name__ == "__main__":
    metadata_filtering()


