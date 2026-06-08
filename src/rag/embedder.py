import hashlib
import numpy as np

class LocalCurriculumEmbedder:
    """
    Generates high-dimensional vector representations of concepts
    to populate OpenSearch / Milvus or Postgres pgvector schemas.
    Runs locally with $0 API overhead.
    """
    
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def generate_concept_embedding(self, concept_text: str) -> list:
        """
        Generates a 1024-dimensional concept vector.
        Uses a deterministic, unit-normalized hash vector as a fast, local, zero-cost fallback.
        """
        # Create deterministic pseudo-random embedding vector
        hash_object = hashlib.sha256(concept_text.encode('utf-8'))
        seed = int(hash_object.hexdigest(), 16) % (2**32 - 1)
        
        rng = np.random.default_rng(seed)
        raw_vector = rng.standard_normal(self.dimension)
        
        # Unit normalization (L2 norm)
        norm = np.linalg.norm(raw_vector)
        if norm == 0:
            normalized_vector = raw_vector
        else:
            normalized_vector = raw_vector / norm
            
        return normalized_vector.tolist()

concept_embedder = LocalCurriculumEmbedder()
