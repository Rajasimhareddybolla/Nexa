# compare.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_similarity(emb1: list[float], emb2: list[float]) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb1 (list[float]): First embedding.
        emb2 (list[float]): Second embedding.
    
    Returns:
        float: Similarity score (0 to 1).
    """
    emb1_array = np.array(emb1).reshape(1, -1)
    emb2_array = np.array(emb2).reshape(1, -1)
    return cosine_similarity(emb1_array, emb2_array)[0][0]