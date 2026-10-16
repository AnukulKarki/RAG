from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()

core_version = version("langchain-core")
graph_version = version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

print(f"langchain-core Versions : {core_version}")
print(f"langgraph version : {graph_version}")

def main():
    #model= openai/gpt-oss-20b have more minute token limit.
    #model = llama-3.3-70b-versatile better response
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
    response = llm.invoke("Say 'Setup complete!' in one word")
    print(f"Response from ChatOpenAI: {response}")
    

if __name__ == "__main__":
    main()
