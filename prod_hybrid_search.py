from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import  BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports gigabit speeds and advacne QoS features.",
        metadata = {'type':'product'}
    ),
    Document(
        page_content='For network connectivity issues, first check the ethernet cable and router status lights.',
        metadata = {'type':'troubleshooting'}
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server rejected the connection. Check firewall settings.",
        metadata = {'type':'error'}
    ),
    Document(
        page_content="The authentication process requires valid credentials. Use OAuth2 for secure API access.",
        metadata = {"type":"auth"}
    ),
    Document(
        page_content="Router configuration guide: Access the admin panel at 192.168.1.1 to modify settings.",
        metadata = {"type":"config"}
    ),
    Document(
        page_content="WCAG 2.1 compliance requires all images to have alt text and sufficient color contrast.",
        metadata = {"type":"compliance"}
    )
]

print(f"Loaded {len(documents)} documents")

vectorstore = Chroma.from_documents(
    documents,
    embeddings,
    collection_name = "hybrid_Test"
)

vector_retriever = vectorstore.as_retriever(search_kwargs ={"k":3})

print('Vector retriever ready')

bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=3
)

print("BM25 retriever ready")


ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5,0.5]
)

def test_query(query,name,retriever):
    '''Test a query and shwo results'''
    results = retriever.invoke(query)
    print(f'\n{name} - Query: \"{query}\"')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + "..."
        print(f" {i+1}, {preview}")

test_queries = [
    "SKU-7742X specifications", #Exact product code
    "E_CONN_REFUSED error",     #error code
    "How do I authenticate?",   #semantic question
    "WCAG compliance",      #acronym   
    "router configuration", #general semantic
]

for query in test_queries:
    print("="*60)

    vector_results = test_query(query, "VECTOR", vector_retriever)

    bm25_results = test_query(query, "BM25", bm25_retriever)

    hybrid_results = test_query(query, "HYBRID", ensemble_retriever)

def hybrid_retrieve(query, retrievers, weights, k = 3, rrf_k=60):
    """combine multiple retirevers using weighted reciprocal rank fusion."""
    doc_scores = {} #page_content -> (score, doc)

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)
        for rank,doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight * (1.0 / (rank + rrf_k))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score, doc)
            else:
                doc_scores[key] = (rrf_score, doc)

    sorted_docs = sorted(doc_scores.value(), key=lambda x:x[0], reverse = True)

    return [doc for _, doc in sorted_docs[:k]]

