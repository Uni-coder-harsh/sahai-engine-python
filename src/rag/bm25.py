import math
import re
from typing import List, Dict, Tuple

class BM25Retriever:
    """
    Pure Python implementation of Okapi BM25 for keyword-based sparse search,
    tailored for curriculum node retrieval.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lengths = {}      # Doc ID -> length
        self.doc_term_freqs = {}   # Doc ID -> {term -> count}
        self.doc_frequencies = {}  # term -> document count
        self.idf = {}              # term -> IDF weight
        self.doc_metadata = {}     # Doc ID -> original details dict

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text by lowercasing and splitting into alphanumeric words.
        """
        if not text:
            return []
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        return [word for word in text.split() if len(word) > 1]

    def fit(self, corpus: List[Dict[str, any]]):
        """
        Fits BM25 index on a corpus list.
        Each item in the corpus must be a dict containing:
        - 'id': unique document identifier (e.g. node_id)
        - 'text': text content to index (e.g. concept_name)
        - 'metadata': dict containing original document details (optional)
        """
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            return
            
        total_len = 0
        self.doc_lengths = {}
        self.doc_term_freqs = {}
        self.doc_frequencies = {}
        self.doc_metadata = {}
        
        # Count term frequencies and document lengths
        for doc in corpus:
            doc_id = doc['id']
            text = doc['text']
            self.doc_metadata[doc_id] = doc.get('metadata', {})
            
            tokens = self._tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)
            total_len += len(tokens)
            
            term_counts = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1
            self.doc_term_freqs[doc_id] = term_counts
            
            # Document frequency (number of docs containing the term)
            for term in term_counts:
                self.doc_frequencies[term] = self.doc_frequencies.get(term, 0) + 1
                
        self.avg_doc_len = total_len / self.corpus_size
        
        # Calculate IDF for each term
        for term, df in self.doc_frequencies.items():
            # Standard Okapi BM25 IDF formula with smoothing to avoid negative weights
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def score_query(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Scores all documents against a query string.
        Returns a list of tuples: (doc_id, score) sorted by score descending.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []
            
        scores = {}
        for doc_id, term_counts in self.doc_term_freqs.items():
            score = 0.0
            doc_len = self.doc_lengths[doc_id]
            
            for token in query_tokens:
                if token not in term_counts:
                    continue
                
                tf = term_counts[token]
                idf_val = self.idf.get(token, 0.0)
                
                # Okapi BM25 formula
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf_val * (numerator / denominator)
                
            if score > 0.0:
                scores[doc_id] = score
                
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:limit]
