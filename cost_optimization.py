#Token budgeting
from langchain_groq import ChatGroq
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()


class TokenBudget:

    def __init__(self, max_tokens_per_requests:int = 4000):
        self.max_tokens_per_requests = max_tokens_per_requests
        self.usage = {
            "total_input":0,
            "total_output":0,
            "requests":0
        }

    def estimate_tokens(self, text:str) -> int:
        """Rough token estimation (actual would use tiktoken)."""
        return int(len(text.split()) * 1.3)

    def check_budget(self, text:str) -> tuple[bool, int]:
        """Check if the request is within the budget"""
        tokens = self.estimate_tokens(text)
        return tokens <= self.max_tokens_per_requests, tokens

    def record_usage(self, input_tokens: int, output_tokens:int):
        """Record token usage"""
        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["requests"] += 1

    def get_stats(self) -> dict:
        return{
            **self.usage,
            "total_tokens":self.usage["total_input"] + self.usage["total_output"],
            "avg_per_request":(
                (self.usage["total_input"] + self.usage["total_output"])/max(self.usage["requests"], 1)
            )
        }

class BudgetedLLM:
    """LLM with token budgeting."""

    def __init__ (self, max_tokens:int=4000):
        self.llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
        self.budget = TokenBudget(max_tokens_per_requests=max_tokens)

    @traceable(name="budgeted_invoke")
    def invoke(self, query:str) -> str:
        #check budget
        within_budget, tokens = self.budget.check_budget(query)
        if not within_budget:
            raise ValueError(
                f"Query exceeds token budget: {tokens} > {self.budget.max_tokens_per_requests}"
            )
        #execute
        response = self.llm.invoke(query)
        result = response.content

        output_tokens = self.budget.estimate_tokens(result)

        self.budget.record_usage(tokens, output_tokens)

        return result


    def get_stats(self) -> dict:
        return self.budget.get_stats()

def demo_token_budgeting():
    """Demonstrate token budgeting"""

    llm = BudgetedLLM(max_tokens=100)

    queries = [
        "What is AI?",
        "Explain " + "very " * 100 + "complex topic"
    ]

    print("Token Budgeting Demo:")

    for query in queries:
        try:
            result = llm.invoke(query)
            print(f"{query[:40]}... -> {result[:30]} ...")
        except ValueError as e:
            print(f"Failed!!! {query[:40]}... -> {e}")

    print(f"\n Usage: {llm.get_stats()}")

if __name__ == "__main__":
    demo_token_budgeting()

    