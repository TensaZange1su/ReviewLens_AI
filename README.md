# ReviewLens AI

### Multilingual Aspect-Based Sentiment Analysis and RAG-Based Review Intelligence

<p align="center">
  <strong>Turning unstructured customer reviews into actionable insights.</strong>
</p>

ReviewLens AI is a multilingual review intelligence system that combines Aspect-Based Sentiment Analysis (ABSA), Transformer-based classification, semantic search, and Retrieval-Augmented Generation (RAG).

The system analyzes customer reviews in English, Russian, and German, identifies sentiment associated with specific aspects, retrieves semantically relevant reviews, and generates evidence-based analytical summaries using a Large Language Model.

---

## Project Overview

Traditional sentiment analysis usually assigns one sentiment label to an entire review. However, customer feedback often contains different opinions about multiple aspects of the same product or service.

For example:

> "The food was excellent, but the service was very slow."

A traditional classifier may assign one overall sentiment to this review. ReviewLens AI aims to identify both opinions separately:

| Aspect | Sentiment |
|---|---|
| Food | Positive |
| Service | Negative |

The project extends this functionality with multilingual semantic retrieval and RAG-based review analysis.

## Key Features

- **Multilingual ABSA:** Aspect-level sentiment and category prediction.
- **Transformer Classification:** Fine-tuned XLM-RoBERTa multi-task model.
- **Semantic Search:** Multilingual sentence embeddings and FAISS vector search.
- **Cross-Lingual Retrieval:** Search for semantically similar reviews across languages.
- **RAG-Based Analysis:** Generate answers using retrieved customer reviews as context.
- **LLM Integration:** Local Qwen3 8B model through Ollama.
- **Interactive Interface:** Streamlit application for user-facing review analysis.

## Supported Languages and Domains

**Languages**
- English
- Russian
- German

**Domains**
- Restaurant
- Hotel
- Laptop

## System Architecture

```text
                   CUSTOMER REVIEWS
                          |
                          v
                  TEXT PREPROCESSING
                          |
             +------------+------------+
             |                         |
             v                         v
       XLM-RoBERTa              Sentence Transformer
             |                         |
             v                         v
     ABSA Classification           Embeddings
             |                         |
       +-----+------+                  v
       |            |                FAISS
       v            v                  |
   Sentiment     Category              v
                                  Top-K Reviews
                                       |
                                       v
                                  RAG Context
                                       |
                                       v
                                    Qwen3
                                       |
                                       v
                              Analytical Summary
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| NLP Model | XLM-RoBERTa |
| Transformer Framework | Hugging Face Transformers |
| Baseline Models | TF-IDF, Logistic Regression |
| Sentence Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Generative AI | Qwen3 8B |
| LLM Runtime | Ollama |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Evaluation | Scikit-learn |

---

## Dataset

The project uses two datasets:

### MAMS

Used for the English ABSA baseline and initial experiments.

### M-ABSA

Used as the main multilingual dataset.

The selected subset contains:

| Dataset characteristic | Value |
|---|---:|
| Aspect annotations | 27,582 |
| Unique reviews | 17,795 |
| Aspect categories | 153 |
| Languages | 3 |
| Domains | 3 |
| Sentiment classes | 3 |

Sentiment labels:
- Positive
- Neutral
- Negative

Data preparation includes annotation parsing, missing-value handling, invalid annotation filtering, class distribution analysis, and review-level split leakage correction.

The same review text is kept within a single train, validation, or test split.

---

## Machine Learning and NLP Pipeline

### 1. Baseline Models

Classical machine-learning baselines were developed using:

- TF-IDF vectorization
- Unigram and bigram features
- Logistic Regression
- Class-weight balancing

Separate experiments were conducted for sentiment and aspect-category classification.

### 2. Multilingual ABSA Model

The main model is based on `xlm-roberta-base`.

A custom multi-task architecture uses a shared Transformer encoder and two classification heads:

```text
                 Input Text + Aspect
                         |
                         v
                   XLM-RoBERTa
                         |
                   Shared Encoder
                         |
                 Pooled Representation
                         |
                 +-------+-------+
                 |               |
                 v               v
          Sentiment Head    Category Head
                 |               |
                 v               v
          3 Sentiment       153 Categories
             Classes
```

The model predicts:
- Sentiment polarity: positive, neutral, negative.
- Aspect category: one of 153 categories.

The model is trained using a combined multi-task classification loss.

### 3. Semantic Retrieval

The retrieval component uses:

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Review embeddings are normalized and indexed using FAISS with inner-product similarity.

This enables efficient semantic search and cross-lingual retrieval.

### 4. Retrieval-Augmented Generation

The RAG pipeline combines semantic retrieval with a local LLM.

```text
User Query
    |
    v
Multilingual Embedding
    |
    v
FAISS Retrieval
    |
    v
Relevant Customer Reviews
    |
    v
Context Construction
    |
    v
Qwen3 8B
    |
    v
Structured Analytical Answer
```

The retrieved reviews provide contextual evidence for the generated response.

The project also compares two approaches:

**LLM-only**

```text
Query -> Qwen3 -> Answer
```

**RAG**

```text
Query -> FAISS -> Retrieved Reviews -> Qwen3 -> Answer
```

---

## Model Evaluation

### Baseline vs. XLM-RoBERTa

| Metric | Baseline | XLM-RoBERTa |
|---|---:|---:|
| Sentiment Accuracy | 79.71% | 90.13% |
| Sentiment Macro-F1 | 58.58% | 60.47% |
| Category Accuracy | 37.56% | 62.24% |
| Category Macro-F1 | 16.94% | 17.37% |

### Multilingual Sentiment Evaluation

| Language | Accuracy | Macro-F1 |
|---|---:|---:|
| English | 90.03% | 59.54% |
| Russian | 90.16% | 60.93% |
| German | 90.17% | 60.89% |

### Key Findings

- XLM-RoBERTa achieved 90.13% sentiment accuracy.
- Sentiment performance remained relatively consistent across the three languages.
- Category classification remains challenging due to the large number of categories.
- Neutral sentiment is the main weakness of the current model.

The neutral class achieved approximately 2.3% recall on the test set, indicating substantial difficulty distinguishing neutral opinions from positive and negative sentiment.

**Note:** Retrieval similarity measures the relevance of retrieved reviews, not the factual correctness or quality of generated answers. These should be evaluated separately.

---

## Interactive Application

ReviewLens AI includes a Streamlit frontend designed to provide an interactive interface for the underlying ML and NLP components.

The application is organized around three main capabilities:

### Review Analyzer
Allows users to submit review text and an aspect to obtain sentiment and category predictions from the trained ABSA model.

### Semantic Search
Allows users to search for similar customer reviews using multilingual embeddings and the FAISS index.

### AI Review Assistant
Uses retrieved reviews as context for Qwen3 to generate structured analytical responses.

The frontend connects to locally stored model artifacts and the local Ollama runtime.

---

## Project Structure

```text
ReviewLens-AI/
│
├── app/
│   └── streamlit_app.py
│
├── assets/
│   └── logo.png
│
├── data/
│   ├── raw/
│   │   ├── mams/
│   │   └── m_absa/
│   │
│   └── processed/
│       ├── m_absa_selected.csv
│       ├── reviews.csv
│       ├── baseline_results.csv
│       └── retrieval_examples.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_absa_baseline.ipynb
│   ├── 03_transformer_training.ipynb
│   ├── 04_multilingual_evaluation.ipynb
│   ├── 05_embeddings_and_retrieval.ipynb
│   ├── 06_rag_evaluation.ipynb
│   └── 07_rag_llm_evaluation.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   ├── retrieval/
│   ├── rag/
│   └── pipeline.py
│
├── models/
│   ├── absa/
│   │   └── xlmr_multitask/
│   └── embeddings/
│
├── vector_store/
│   ├── index.faiss
│   └── metadata.pkl
│
├── configs/
│   └── config.yaml
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd ReviewLens-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If FAISS is not included in the requirements:

```bash
python -m pip install faiss-cpu
```

### 4. Prepare model artifacts

The application requires the trained XLM-RoBERTa model, tokenizer, embedding model, and FAISS index.

Expected artifacts include:

```text
models/absa/xlmr_multitask/
    model.pt
    tokenizer.json
    config.json

vector_store/
    index.faiss
    metadata.pkl
```

Large model weights and generated vector indexes may be excluded from GitHub. They must be obtained from the training pipeline or a separate artifact storage location.

### 5. Configure Ollama

Install Ollama and pull the required model:

```bash
ollama pull qwen3:8b
```

Start Ollama:

```bash
ollama serve
```

Verify that the model is available:

```bash
ollama list
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local application:

```text
http://localhost:8501
```

---

## Evaluation Methodology

The project uses different evaluation approaches for different components.

| Component | Evaluation |
|---|---|
| Sentiment Classification | Accuracy, Precision, Recall, Macro-F1 |
| Category Classification | Accuracy, Precision, Recall, Macro-F1 |
| Multilingual Evaluation | Metrics by language and domain |
| Semantic Retrieval | Top-K similarity |
| RAG | Evidence relevance and response analysis |
| LLM Comparison | LLM-only vs. RAG |

Confusion matrices and error analysis are used to investigate classification weaknesses.

---

## Limitations

The current implementation has several limitations:

- Only three languages and three domains are included.
- The sentiment dataset has significant class imbalance.
- Neutral sentiment detection requires improvement.
- Aspect-category classification is difficult due to the large label space.
- Retrieval quality depends on embedding quality and the indexed dataset.
- RAG output quality depends on the relevance of retrieved evidence.
- The current ABSA model requires an aspect as input; automatic aspect extraction is not yet part of the classifier.

## Future Improvements

- Improve neutral sentiment detection.
- Experiment with class balancing and loss functions.
- Improve aspect-category classification.
- Add automatic aspect extraction.
- Introduce retrieval reranking.
- Expand language and domain coverage.
- Develop a more advanced interactive dashboard.
- Add systematic RAG evaluation and hallucination checks.
- Integrate real-world customer review sources.

---

## Project Deliverables

- Multilingual ABSA dataset and preprocessing pipeline.
- Classical ML baseline models.
- Fine-tuned XLM-RoBERTa multi-task classifier.
- Multilingual embedding and FAISS retrieval pipeline.
- RAG pipeline with Qwen3 8B.
- Evaluation notebooks and experimental results.
- Streamlit interactive application.

---

## Final Vision

ReviewLens AI aims to bridge the gap between traditional sentiment classification and evidence-based customer feedback intelligence.

By combining multilingual NLP, aspect-level sentiment analysis, semantic retrieval, and generative AI, the system transforms unstructured reviews into structured and actionable insights.

**ReviewLens AI — Understand every aspect of customer feedback.**
