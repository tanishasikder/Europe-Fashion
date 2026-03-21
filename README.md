
---

# Fashion Intelligence Pipeline (End-to-End ML System)

## Overview

This project is an end-to-end machine learning pipeline designed for fashion business owners to make data-driven decisions.

Users can:

* Upload clothing images
* Input product-related parameters
* Receive predictions (e.g., pricing insights, profit margins)
* Generate AI-powered business reports

The system combines **computer vision, classical ML, and LLMs** into a unified pipeline.

---

## Architecture

### 1. Image Classification (PyTorch)

* Users upload a clothing image
* A trained PyTorch model classifies the clothing type (e.g., shirt, dress, etc.)
* Model weights are saved and loaded for inference

### 2. Structured Prediction Model (Scikit-learn)

* Takes user inputs such as:

  * Original price
  * Catalog price
  * Clothing size
  * Predicted clothing type (from CV model)
* Outputs:

  * Estimated profit margins
  * Pricing insights
  * Other business metrics

### 3. API Layer (FastAPI)

* Serves as the central interface
* Handles:

  * Image uploads
  * Model inference calls
  * Data processing
* Integrates both ML models into a unified pipeline

### 4. Retrieval-Augmented Generation (RAG)

* Uses a vector database to store relevant fashion/business knowledge
* Enhances LLM responses with context
* Generates detailed reports based on predictions

### 5. LLM Report Generation

* Produces business insights such as:

  * Market positioning
  * Pricing strategy recommendations
  * Profitability analysis

### 6. Data Storage

* **SQLite3**

  * Stores user inputs
  * Stores past predictions
* **Vector Database**

  * Stores embeddings for RAG

### 7. Experiment Tracking (MLflow)

* Tracks:

  * Model performance
  * Experiments
  * Parameters
* Helps with reproducibility and versioning

---

## Tech Stack

* **Machine Learning:** Scikit-learn
* **Deep Learning:** PyTorch
* **Backend:** FastAPI
* **Frontend:** HTML
* **Database:** SQLite3
* **Vector DB:** (e.g., FAISS / Chroma)
* **Experiment Tracking:** MLflow
* **LLM Integration:** RAG + API-based LLM

---

##  End-to-End Pipeline Flow

1. User uploads clothing image
2. PyTorch model predicts clothing type
3. User inputs product parameters
4. Scikit-learn model predicts business metrics
5. Results stored in SQLite
6. Relevant context retrieved via vector DB
7. LLM generates a detailed report
8. Final output returned via API + frontend

---
