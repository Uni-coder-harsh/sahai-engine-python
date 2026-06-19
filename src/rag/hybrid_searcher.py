from typing import List, Dict, Tuple
from rag.vector_store import vector_store
from rag.bm25 import BM25Retriever
from utils.logger import logger

class HybridSearcher:
    """
    Orchestrates dense vector search and sparse BM25 search, combining their 
    ranks using Reciprocal Rank Fusion (RRF) to achieve state-of-the-art retrieval accuracy.
    """
    
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever()
        self.is_initialized = False

    def initialize_bm25(self, pg_conn):
        """
        Loads all concepts from PostgreSQL and builds the BM25 index.
        """
        logger.info("Initializing BM25 retriever index from Postgres concept nodes...")
        try:
            with pg_conn.cursor() as cur:
                cur.execute("SELECT node_id, concept_name, difficulty_baseline FROM concept_nodes;")
                rows = cur.fetchall()
            
            corpus = []
            for row in rows:
                node_id, concept_name, difficulty = row
                corpus.append({
                    "id": node_id,
                    "text": concept_name,
                    "metadata": {
                        "concept_name": concept_name,
                        "difficulty": float(difficulty)
                    }
                })
            
            self.bm25.fit(corpus)
            self.is_initialized = True
            logger.info(f"Successfully indexed {len(corpus)} concepts for BM25 search.")
        except Exception as e:
            logger.error(f"Failed to initialize BM25 retriever: {e}")
            self.is_initialized = False

    def search(self, pg_conn, query_text: str, limit: int = 5) -> List[Dict[str, any]]:
        """
        Executes hybrid retrieval by running vector search and BM25 search,
        then combining ranks using Reciprocal Rank Fusion (RRF).
        """
        if not self.is_initialized:
            self.initialize_bm25(pg_conn)
            
        # 1. Run Vector Dense Search
        # Returns list of (node_id, concept_name, score)
        vector_results = vector_store.query_dense_similarity(pg_conn, query_text, limit=limit * 3)
        
        # 2. Run BM25 Sparse Search
        # Returns list of (node_id, score)
        bm25_results = []
        if self.is_initialized:
            try:
                bm25_results = self.bm25.score_query(query_text, limit=limit * 3)
            except Exception as e:
                logger.error(f"BM25 query failed: {e}")
        
        # 3. Perform Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        # Map node_id -> concept_name for convenient lookup later
        concept_names = {}
        for node_id, name, _ in vector_results:
            concept_names[node_id] = name
            
        # Initialize concept_names from BM25 metadata if not already present
        if self.is_initialized:
            for node_id, _ in bm25_results:
                if node_id not in concept_names:
                    meta = self.bm25.doc_metadata.get(node_id, {})
                    concept_names[node_id] = meta.get("concept_name", node_id)
        
        # Process Vector rank scores
        for rank, (node_id, _, _) in enumerate(vector_results):
            # rank is 0-indexed, RRF requires 1-indexed rank
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            
        # Process BM25 rank scores
        for rank, (node_id, _) in enumerate(bm25_results):
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            
        # Sort by RRF score descending
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        final_results = []
        for node_id, score in sorted_rrf[:limit]:
            name = concept_names.get(node_id, node_id)
            final_results.append({
                "node_id": node_id,
                "concept_name": name,
                "rrf_score": score
            })
            
        return final_results

hybrid_searcher = HybridSearcher()
