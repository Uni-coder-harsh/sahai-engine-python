import pytest
import base64
import numpy as np
from unittest import mock
from unittest.mock import MagicMock
from rag.normalizer import code_normalizer
from rag.chunker import code_chunker
from rag.vector_store import vector_store
from rag.bm25 import BM25Retriever
from rag.hybrid_searcher import hybrid_searcher
from models.ocr_handler import ocr_handler
from models.bayesian_network import update_bayesian_network

# --- RAG Normalizer Tests ---

def test_code_normalizer_removes_comments_and_docstrings():
    python_code = """
# This is a comment
def add(a, b):
    \"\"\"This is a docstring.\"\"\"
    return a + b  # inline comment
"""
    cleaned = code_normalizer.remove_python_comments_and_docstrings(python_code)
    assert "# This is a comment" not in cleaned
    assert "This is a docstring" not in cleaned
    assert "inline comment" not in cleaned
    assert "def add(a, b):" in cleaned
    assert "return a + b" in cleaned

def test_code_normalizer_ocr_cleanup():
    ocr_noise = "..__def my_func():\n   print('hello')"
    cleaned = code_normalizer.clean_extracted_ocr(ocr_noise)
    assert "def my_func():" in cleaned
    assert "print('hello')" in cleaned

# --- RAG Chunker Tests ---

def test_code_chunker_splits_blocks():
    python_code = "\n".join([f"def func_{i}():\n    pass" for i in range(15)])
    chunks = code_chunker.chunk_python_code(python_code)
    assert len(chunks) > 1
    assert "def func_0():" in chunks[0]["text"]
    assert chunks[0]["start_line"] == 1

# --- RAG BM25 Tests ---

def test_bm25_fits_and_retrieves():
    corpus = [
        {"id": "PY_SYNTAX_01", "text": "Basic indentation and comments syntax", "metadata": {}},
        {"id": "PY_OOP_01", "text": "Class definition constructor and instance objects", "metadata": {}},
        {"id": "PY_CONTROL_01", "text": "Conditional statement if elif else blocks", "metadata": {}}
    ]
    retriever = BM25Retriever()
    retriever.fit(corpus)
    
    # Query matching class definitions
    results = retriever.score_query("class definition instance", limit=2)
    assert len(results) >= 1
    assert results[0][0] == "PY_OOP_01"
    assert results[0][1] > 0.0

# --- RAG Hybrid Searcher / Vector Tests ---

def test_rrf_scoring_order():
    # Construct a HybridSearcher instance manually to mock outputs
    from rag.hybrid_searcher import HybridSearcher
    searcher = HybridSearcher()
    
    # Mock vector query results and BM25 results
    mock_vector = [("PY_OOP_01", "Class Definition", 0.9), ("PY_SYNTAX_01", "Basic Indentation", 0.7)]
    mock_bm25 = [("PY_SYNTAX_01", 1.5), ("PY_CONTROL_01", 1.2)]
    
    with mock.patch('rag.vector_store.PostgresVectorStore.query_dense_similarity', return_value=mock_vector):
        with mock.patch.object(searcher.bm25, 'score_query', return_value=mock_bm25):
            searcher.is_initialized = True
            searcher.bm25.doc_metadata = {
                "PY_OOP_01": {"concept_name": "Class Definition"},
                "PY_SYNTAX_01": {"concept_name": "Basic Indentation"},
                "PY_CONTROL_01": {"concept_name": "If Statements"}
            }
            
            merged = searcher.search(MagicMock(), "indentation class", limit=3)
            # RRF should combine ranks:
            # PY_SYNTAX_01 is rank 2 in vector (score = 1/(60+2)) and rank 1 in BM25 (score = 1/(60+1)) -> ~0.0325
            # PY_OOP_01 is rank 1 in vector (score = 1/(60+1)) -> ~0.0163
            # PY_CONTROL_01 is rank 2 in BM25 (score = 1/(60+2)) -> ~0.0161
            # PY_SYNTAX_01 should be the top result
            assert merged[0]["node_id"] == "PY_SYNTAX_01"
            assert merged[1]["node_id"] == "PY_OOP_01"
            assert merged[2]["node_id"] == "PY_CONTROL_01"

# --- OCR Handler Grader Tests ---

@mock.patch('pytesseract.image_to_string')
def test_extract_code_from_image(mock_tesseract):
    mock_tesseract.return_value = "def add_nums(x, y):\n  return x + y\n"
    # Simple base64 for a 1x1 black pixel GIF
    base64_gif = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    
    extracted = ocr_handler.extract_code_from_image(base64_gif)
    assert "def add_nums" in extracted
    assert "return x + y" in extracted

@mock.patch('utils.llm_client.MultiProviderLLMClient.request_json')
def test_evaluate_logic_via_llm_correct(mock_request):
    mock_request.return_value = {
        "is_correct": True,
        "logical_flaw_explanation": None,
        "failed_node_id": None
    }
    
    context = {"node_id": "PY_OOP_01", "concept_name": "Class Definition"}
    result = ocr_handler.evaluate_logic_via_llm(
        extracted_text="class Car:\n  pass",
        question_context=context,
        telemetry_metrics={"time_spent_sec": 45},
        allowed_node_ids=["PY_OOP_01", "PY_SYNTAX_05"]
    )
    assert result["is_correct"] is True
    assert result["failed_node_id"] is None

@mock.patch('utils.llm_client.MultiProviderLLMClient.request_json')
def test_evaluate_logic_via_llm_incorrect_hallucination_handling(mock_request):
    mock_request.return_value = {
        "is_correct": False,
        "logical_flaw_explanation": "Missing constructor call.",
        # Hallucinated node_id not in allowed_node_ids
        "failed_node_id": "PY_CLASS_INNER_99"
    }
    
    context = {"node_id": "PY_OOP_01", "concept_name": "Class Definition"}
    result = ocr_handler.evaluate_logic_via_llm(
        extracted_text="class Car:\n  def __init__(self):\n    pass",
        question_context=context,
        telemetry_metrics={"time_spent_sec": 120},
        # Strict allowed node IDs
        allowed_node_ids=["PY_OOP_01", "PY_OOP_05", "PY_SYNTAX_05"]
    )
    assert result["is_correct"] is False
    # The handler must resolve the hallucinated ID to the fallback primary node_id
    assert result["failed_node_id"] == "PY_OOP_01"

# --- Bayesian updates Integration Tests ---

@mock.patch('models.bayesian_network.fetch_or_init_state')
@mock.patch('models.bayesian_network.save_cognitive_state')
@mock.patch('models.bayesian_network.propagate_updates_up_dag')
def test_update_bayesian_network(mock_propagate, mock_save, mock_fetch):
    # Setup mock return values
    mock_fetch.return_value = {
        "user_id": "test_user_id",
        "node_id": "PY_OOP_01",
        "distribution": {"alpha": 2.0, "beta": 2.0},
        "temporal_factors": {
            "last_practiced": "2026-06-19T22:00:00+00:00",
            "forgetting_curve_decay_rate": 0.02
        }
    }
    
    mock_conn = MagicMock()
    mock_mongo = MagicMock()
    mock_redis = MagicMock()
    
    res = update_bayesian_network(
        user_id="test_user_id",
        failed_node_id=None,
        is_correct=True,
        telemetry_metrics={"run_count": 2},
        primary_node_id="PY_OOP_01",
        mongo_db=mock_mongo,
        pg_conn=mock_conn,
        r_client=mock_redis
    )
    
    assert res["success"] is True
    assert res["target_node"] == "PY_OOP_01"
    # Verify DB operations called
    mock_fetch.assert_called_once()
    mock_save.assert_called_once()
    mock_propagate.assert_called_once()
