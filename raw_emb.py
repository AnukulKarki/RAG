from dotenv import load_dotenv
from google import genai
import os

load_dotenv()
def model_compare():
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    chat_repsonse = client.models.generate_content(model="gemini-3.8-flash",contents="What is the capital city of Nepal?")
    print()
    print(chat_repsonse)
    print()
    print(chat_repsonse.text)

    #embedding models

    print("-------------Generating Embedding Vectors ----------------")
    embedding_response = client.models.embed_content(model="gemini-embedding-2", contents="What is my Name?")

    print()
    print()
    vector_values = embedding_response.embeddings[0].values
    print(f"Dimension of Embedding {len(vector_values)}")
    print(f"First 10 vectors: {vector_values[:10]}")
    print()
    print()
    print(embedding_response)


if __name__ == "__main__":
    model_compare()