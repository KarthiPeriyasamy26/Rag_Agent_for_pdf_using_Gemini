from query_data import query_rag
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_community.vectorstores import Chroma


EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""


def test_monopoly_rules():
     assert query_and_validate(
        question="Is the Milvus vector Database is better for Hybrid search ?",
        expected_response="Yes",
    )


def test_ticket_to_ride_rules():
    assert query_and_validate(
        question="What is the smaller chunk size for standard RAG? (Answer with the number only)",
        expected_response="512",
    )


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    # Initialize gemini-2.0-flash model for evaluation
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
    evaluation_results = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results.content.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )