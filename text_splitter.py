from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language
)

from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Sample documents for testing
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''

def recursive_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(SAMPLE_TEXT)


    print(f"Original Length: {len(SAMPLE_TEXT)} Chars")
    print(f"Number of Chunks: {len(chunks)}")
    print(f"Chunk sizes: {[len(c) for c in chunks]} Chars")
    print(f"First Chunk Preview: \n{chunks[0][:200]} ...")

def document_splitter():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document

    loader = PyPDFLoader("./docs/langchain_demo.pdf")
    docs = loader.load()

    print(f"Loaded {len(docs)} documents from pdf")

    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)

    split_docs = splitter.split_documents(docs)

    print(f"Split Into: {len(split_docs)} Chunks")
    print(f"First Chunk Metadata: {split_docs[0].metadata}")
    print(f"First Chunk Content: {split_docs[0].page_content[:200]} ")
    print(f"Last Chunk Metadata: {split_docs[-1].metadata} ...")

def markdown_text_splitter():
    headers_to_split_on = [
        ("#","Header 1"),
        ("##","Header 2"),
        ("###","Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    docs = markdown_splitter.split_text(SAMPLE_TEXT)


    for i, doc in enumerate(docs):
        print(f"\nDocument: {i+1}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content}")
        print("--"*50)

def code_splitter():
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size = 500, chunk_overlap=50)
    chunks = python_splitter.split_text(SAMPLE_CODE)
    print(f"Code Splitter Producd {len(chunks)} Chunks")

    for i, chunk in enumerate(chunks):
        print(f"\n Chunk {i} {len(chunk)} Characters")
        # print(chunk[:150] + "... " if len(chunk)>150 else chunk)
        print(chunk)
        print()


if __name__ == "__main__":
    # recursive_splitter()
    # document_splitter()
    # markdown_text_splitter()
    code_splitter()