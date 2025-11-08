# embed.py
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize embeddings (done once)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}  # Local CPU
)

def get_embedding(text: str) -> list[float]:
    """
    Get vector embedding for the given text.
    
    Args:
        text (str): Input text.
    
    Returns:
        list[float]: Embedding vector.
    """
    return embeddings.embed_query(text)