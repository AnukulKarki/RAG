from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embeddings = DefaultEmbeddingFunction()
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

def basic_emebeddings():
    #single Text
    text = "What is machine learning?"

    single_embedding = embedding_model.embed_query(text)
    # print(np.array(single_embedding).shape)
    print(f"Vector Dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector Norm {np.linalg.norm(single_embedding):.4f}")

def batch_embeddings():
    texts = [
        "What is machine learning",
        "Explain the concept of overfitting in ML.",
        "How are you?"
    ]
    # use embed_documents for batch 
    batch_embeddings = embedding_model.embed_documents(texts)
    for i, emb in enumerate(batch_embeddings):
        print(f"Text {i+1} -- Vector Dimensions: {len(emb)}")
        print(f"Text {i+1} -- First 5 value: {emb[:5]}")
        print(f"Text {i+1} -- Vector Norm: {np.linalg.norm(emb):.4f}")


def similarity_search():
    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = "What programming languages exist?"

    #embved documents
    doc_vector = embedding_model.embed_documents(docs)
    query_vector = embedding_model.embed_query(query)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, document_vector) for document_vector in doc_vector]

    ranked_docs = sorted(zip(docs, similarities), key = lambda x:x[1], reverse=True )

    for doc, score in ranked_docs:
        print(f"Document: {doc}")
        print(f"Similarity Score: {score}")
        print()

    





if __name__ == "__main__":
    # basic_emebeddings()
    # batch_embeddings()
    similarity_search()
