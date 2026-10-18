import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (TextLoader, PyPDFLoader )
from dotenv import load_dotenv

load_dotenv()

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello, this is a sample text file.\nThis file is used to demonstarate the TextLoader.")

    temp_file_path = temp_file.name

    print("Location of the temp file: "+ temp_file_path)

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        print(f"Loaded {len(documents)} document(s)")
        print(f"Content Preview: {documents[0].page_content[:100]}...")
        print(f"Metadata: {documents[0].metadata}")


        # print("This is the documents variable: -----------------")
        # print(documents)
        # print("End of the documents --------------------")
        # for doc in documents:
        #     print("Document Content:")
        #     print(doc)
        #     print("End of Document Content -----")
        #     print(doc.page_content)
    finally:
        os.remove(temp_file_path)


def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from PDF")
    # print(documents)
    for i, doc in enumerate(documents):
        print(f"Document {i+1} Content Preview: {documents[i].page_content[0:100]}")
        print(f"Metadata: {documents[i].metadata}")
        print()
        print()
    



if __name__ == "__main__":
    pdf_loader("./docs/langchain_demo.pdf")