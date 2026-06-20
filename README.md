# SahAI Python Math Engine: Cognitive Inference Worker

This directory contains the Python-based cognitive inference and math engine (`sahai-engine-python`). It performs real-time Bayesian Knowledge Tracing (BKT) graph updates, processes live Random Forest telemetry classifications, handles handwriting transcription OCR, runs hybrid pgvector + BM25 RAG, and executes LLM grading pipelines.

---

## 🧠 Computational Engines & Mathematical Modeling

### 1. Bayesian Knowledge Tracing (BKT) with Priors
Student conceptual mastery is modeled as probability density functions using Beta distributions:
* **Initial Priors**: Set to $\alpha=2.0, \beta=2.0$, creating a balanced probability curve centered at $E[x] = 0.5$. This prevents the graph from fluctuating too rapidly on initial diagnostic responses.
* **Mastery Expectation**:
  $$E[K] = \frac{\alpha}{\alpha + \beta}$$
* **Direct Telemetry Updates**:
  * Correct Response: $\alpha_{new} = \alpha_{old} + 1.0 \times \text{modifier}$
  * Incorrect Response: $\beta_{new} = \beta_{old} + 1.0 \times \text{modifier}$

### 2. Prerequisite DAG Propagation
To avoid expensive database joins, the engine loads the entire prerequisite matrix into Upstash Redis memory (`global_dag` hash) at startup.
* **Success Propagation ($W_{pre}$)**: When a concept is mastered, parent concept states are boosted using the asymmetric prerequisite weight:
  $$\alpha_{parent} = \alpha_{parent} + W_{pre} \times \text{modifier}$$
* **Failure Diagnostics ($W_{diag}$)**: When a concept response fails, parent concept states are penalized using the diagnostic correlation weight:
  $$\beta_{parent} = \beta_{parent} + W_{diag} \times \text{modifier}$$

### 3. Temporal Ebbinghaus Forgetting Decay
Applies Ebbinghaus forgetting updates dynamically, scaling cognitive values over time:
$$\alpha_{decayed} = 2.0 + (\alpha - 2.0) \times e^{-\lambda t}$$
$$\beta_{decayed} = 2.0 + (\beta - 2.0) \times e^{-\lambda t}$$
Where $\lambda$ represents the half-life retention decay constant, and $t$ is the elapsed seconds since the student's last concept engagement.

### 4. Live Random Forest Telemetry Classifiers
Loads three distinct trained Random Forest models (`telemetry_mcq_model.pkl`, `telemetry_code_model.pkl`, `telemetry_ocr_model.pkl`) at startup. Categorizes student inputs into cognitive behavior classes, injecting learning rate modifiers (penalties) into the Bayesian update chain:
* **MCQ Guesses**: 50% update penalty.
* **Code Copy-Paste / Plagiarism**: 50% update penalty.
* **Shotgun Debugging (Rapid compile trials)**: 20% update penalty.
* **Anxious Overworking**: 5% update penalty.

---

## 📂 Submodule Directory Layout

* `src/main.py` - Light HTTP server exposing `/process-telemetry` (synchronous grading and updates) and `/health` checking endpoints.
* `src/config.py` - Directory-traversing environment configs.
* `src/database/` - Postgres (`pg_connector.py`), Upstash Redis, and MongoDB log storage connector clients.
* `src/models/` - BKT updates (`bayesian_network.py`), Groq/OpenAI vision transcription (`ocr_handler.py`), and Tesseract fallbacks.
* `src/rag/` - Chunker, BM25 keyword indexer, pgvector embeddings searcher, and Reciprocal Rank Fusion query blender.
* `tests/` - Math logic (`test_math.py`) and RAG grading integration (`test_ocr_rag.py`) test suites.

---

## 🚀 Execution & Setup Guide

### 1. Requirements
Ensure Python 3.11+ and `tesseract-ocr` are installed on your host system:
```bash
sudo apt-get install tesseract-ocr
```

### 2. Local Setup
Set up your virtual environment and install backend library dependencies:
```bash
# Initialize venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Engine
Start the math inference server (default port: `5000`):
```bash
PYTHONPATH=src python src/main.py
```

### 4. Running Verification Tests
Execute the Pytest suite to validate mathematical boundaries:
```bash
pytest tests/
```
