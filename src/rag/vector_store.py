import numpy as np
from typing import List, Dict, Tuple
from rag.embedder import concept_embedder
from utils.logger import logger

class PostgresVectorStore:
    """
    Manages vector storage and dense retrieval using PostgreSQL's pgvector extension.
    Encapsulates HNSW index maintenance, vector formatting, and cosine-similarity searches.
    """
    
    def __init__(self):
        pass

    def format_vector_for_pg(self, vector: List[float]) -> str:
        """
        Formats a python float list into a pgvector-compatible string (e.g., '[0.1,0.2,...]').
        """
        return "[" + ",".join(map(str, vector)) + "]"

    def populate_database_embeddings(self, pg_conn, force: bool = False):
        """
        Generates and inserts vector embeddings for all curriculum nodes in `concept_nodes` 
        that do not yet have one.
        """
        logger.info("Initializing vector embedding generation for concept nodes...")
        try:
            with pg_conn.cursor() as cur:
                # Get all nodes that need embedding updates
                if force:
                    cur.execute("SELECT node_id, concept_name FROM concept_nodes;")
                else:
                    cur.execute("SELECT node_id, concept_name FROM concept_nodes WHERE vector_embedding IS NULL;")
                
                rows = cur.fetchall()
                if not rows:
                    logger.info("Vector store is up-to-date. No nodes require vectorization.")
                    return
                
                logger.info(f"Generating embeddings for {len(rows)} concept nodes...")
                for node_id, concept_name in rows:
                    # Generate deterministic unit-normalized embedding
                    embedding = concept_embedder.generate_concept_embedding(concept_name)
                    embedding_str = self.format_vector_for_pg(embedding)
                    
                    cur.execute(
                        """
                        UPDATE concept_nodes
                        SET vector_embedding = %s
                        WHERE node_id = %s;
                        """,
                        (embedding_str, node_id)
                    )
                
                pg_conn.commit()
                logger.info(f"Successfully vectorized and committed {len(rows)} nodes to Postgres.")
                
                # Check / create HNSW index for speed at scale (MNC-grade)
                try:
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_concept_nodes_hnsw_cosine 
                        ON concept_nodes USING hnsw (vector_embedding vector_cosine_ops);
                        """
                    )
                    pg_conn.commit()
                    logger.info("Verified/Created pgvector HNSW index for cosine-similarity.")
                except Exception as index_err:
                    pg_conn.rollback()
                    logger.warn(f"Failed to create HNSW index (could be standard Postgres limitations): {index_err}. Continuing with fallback sequence.")
        except Exception as e:
            pg_conn.rollback()
            logger.error(f"Error populating database concept embeddings: {e}")

    def query_dense_similarity(self, pg_conn, query_text: str, limit: int = 10) -> List[Tuple[str, str, float]]:
        """
        Queries the concept nodes table using cosine distance.
        Returns a list of tuples: (node_id, concept_name, similarity_score).
        Similarity score is computed as (1 - distance) from cosine distance operator <=>.
        """
        query_vector = concept_embedder.generate_concept_embedding(query_text)
        query_vector_str = self.format_vector_for_pg(query_vector)
        
        try:
            with pg_conn.cursor() as cur:
                # pgvector cosine distance operator <=>
                # 1 - distance gives the standard cosine similarity score
                cur.execute(
                    """
                    SELECT node_id, concept_name, 1 - (vector_embedding <=> %s::vector) as similarity
                    FROM concept_nodes
                    WHERE vector_embedding IS NOT NULL
                    ORDER BY vector_embedding <=> %s::vector ASC
                    LIMIT %s;
                    """,
                    (query_vector_str, query_vector_str, limit)
                )
                rows = cur.fetchall()
                # Cast results to native types
                return [(row[0], row[1], float(row[2])) for row in rows]
        except Exception as e:
            logger.error(f"Dense vector query failed: {e}. Executing in-memory fallback...")
            return self._in_memory_fallback_query(pg_conn, query_vector, limit)

    def _in_memory_fallback_query(self, pg_conn, query_vector: List[float], limit: int) -> List[Tuple[str, str, float]]:
        """
        In-memory fallback cosine similarity in case of db-level pgvector errors.
        """
        try:
            with pg_conn.cursor() as cur:
                cur.execute("SELECT node_id, concept_name, vector_embedding FROM concept_nodes WHERE vector_embedding IS NOT NULL;")
                rows = cur.fetchall()
                
            results = []
            qv = np.array(query_vector)
            qv_norm = np.linalg.norm(qv)
            
            for node_id, name, emb_str in rows:
                if not emb_str:
                    continue
                # Parse pgvector string '[val1,val2,...]'
                vals = [float(x) for x in emb_str.strip('[]').split(',')]
                ev = np.array(vals)
                ev_norm = np.linalg.norm(ev)
                
                if qv_norm == 0 or ev_norm == 0:
                    similarity = 0.0
                else:
                    similarity = np.dot(qv, ev) / (qv_norm * ev_norm)
                    
                results.append((node_id, name, float(similarity)))
                
            # Sort by similarity descending
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:limit]
        except Exception as e:
            logger.error(f"In-memory vector query fallback failed: {e}")
            return []

vector_store = PostgresVectorStore()
