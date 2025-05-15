from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
def get_embedding_function():
    # Ensure you have your Google API key set as an environment variable
    # or pass it directly.
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    return embeddings

if __name__ == "__main__":
    # Example usage (requires GOOGLE_API_KEY to be set)
    try:
        embedding_function = get_embedding_function()
        text = "This is a sample text to embed."
        embeddings = embedding_function.embed_query(text)
        print(f"Embeddings for '{text}':")
        print(f"Shape: {len(embeddings)}")
        print(f"First 5 dimensions: {embeddings[:5]}")
    except ValueError as e:
        print(f"Error: {e}")