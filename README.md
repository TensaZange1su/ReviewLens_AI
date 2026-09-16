# ReviewLens AI

## Multilingual Aspect-Based Sentiment Analysis & Review Intelligence

ReviewLens AI is an NLP project for multilingual customer review analysis using Aspect-Based Sentiment Analysis (ABSA), transformer models, multilingual sentence embeddings, semantic retrieval, Retrieval-Augmented Generation (RAG), and a local Large Language Model (LLM).

The system works with three languages:

- English
- Russian
- German

Selected domains:

- Restaurant
- Hotel
- Laptop

---

## Project Goal

The goal of ReviewLens AI is to build an end-to-end review intelligence system that can:

- analyze customer sentiment;
- identify aspect categories;
- work with multilingual reviews;
- retrieve semantically relevant reviews;
- provide evidence-based answers using retrieved customer feedback;
- compare a standalone LLM with a Retrieval-Augmented Generation pipeline.

The project combines classical NLP, transformer-based classification, semantic search, vector retrieval, and local LLM generation.

---

## System Architecture

The complete RAG pipeline is:

Customer Query
        |
        v
Multilingual Sentence Embedding
        |
        v
FAISS Semantic Retrieval
        |
        v
Top-K Relevant Reviews
        |
        v
Context Construction
        |
        v
Qwen LLM
        |
        v
Evidence-based Answer


A separate LLM-only baseline is also implemented:

Customer Query
        |
        v
Qwen LLM
        |
        v
Answer

The LLM-only pipeline does not have access to the review database.

---

## Main Components

### 1. Multilingual Aspect-Based Sentiment Analysis

The ABSA component predicts information associated with a specific aspect of a customer review.

The model predicts:

- Sentiment polarity
- Aspect category

Sentiment classes:

- Positive
- Neutral
- Negative

The input consists of:

Review text + target aspect

Example:

Review:
"The food was excellent but the service was slow."

Aspect:
"service"

Possible output:

Sentiment:
Negative

Aspect category:
Service

---

## 2. Datasets

Two datasets are used in the project.

### MAMS

MAMS is used as an English ABSA baseline dataset.

It provides aspect-based sentiment annotations and is used for baseline experimentation.

### M-ABSA

M-ABSA is the main multilingual dataset used for the project.

The selected subset contains:

Languages:

- English
- Russian
- German

Domains:

- Restaurant
- Hotel
- Laptop

Dataset statistics after preprocessing:

- Aspect annotations: 27,582
- Unique reviews: 17,795
- Sentiment classes: 3
- Aspect categories: 153

The original dataset contained reviews distributed across train, validation, and test splits.

During preprocessing, duplicate review texts appearing across different official splits were detected and corrected to prevent train/test leakage.

---

## 3. Data Exploration and Preprocessing

The first notebook performs dataset exploration and preprocessing.

Notebook:

01_data_exploration.ipynb

Main steps:

1. Load M-ABSA data.
2. Select required languages.
3. Select required domains.
4. Parse the original annotation format.
5. Extract:
   - review text;
   - aspect;
   - aspect category;
   - sentiment;
   - language;
   - domain.
6. Remove invalid conflict annotations.
7. Handle missing values.
8. Analyze class distributions.
9. Detect duplicate review texts.
10. Prevent data leakage between train, validation, and test.
11. Save processed datasets.

Processed files include:

data/processed/m_absa_selected.csv

data/processed/reviews.csv

---

## 4. Classical Machine Learning Baseline

Notebook:

02_absa_baseline.ipynb

A classical NLP baseline is implemented using:

- TF-IDF
- Logistic Regression

Two classification tasks are evaluated:

1. Sentiment classification
2. Aspect category classification

The sentiment classifier uses the review text.

The category classifier uses:

Review text + target aspect

This baseline provides a reference point for evaluating the transformer model.

### Baseline Results

Sentiment:

- Validation Macro-F1: 56.85%
- Test Macro-F1: 58.58%
- Test Accuracy: 79.71%

Aspect category:

- Validation Macro-F1: 15.79%
- Test Macro-F1: 16.94%
- Test Accuracy: 37.56%

---

## 5. Transformer-Based ABSA

Notebook:

03_transformer_training.ipynb

The main ABSA model is based on:

XLM-RoBERTa

Model:

xlm-roberta-base

The architecture uses a shared multilingual transformer encoder with two classification heads.

Architecture:

                    XLM-RoBERTa
                         |
                  Shared Encoder
                         |
                 +-------+-------+
                 |               |
                 v               v
        Sentiment Head    Category Head
                 |               |
                 v               v
          3 classes          153 classes

The model receives:

- review text;
- target aspect.

It predicts:

- sentiment;
- aspect category.

### Training Configuration

Model:

xlm-roberta-base

Maximum sequence length:

128

Learning rate:

2e-5

Number of epochs:

3

Training batch size:

8

Gradient accumulation:

2

Weight decay:

0.01

The model was trained using GPU acceleration.

---

## 6. Transformer Evaluation

Notebook:

04_multilingual_evaluation.ipynb

The transformer model is evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1

Evaluation is performed:

- overall;
- by language;
- by domain;
- by sentiment class.

Confusion matrices and error analysis are also included.

### Overall Test Results

Sentiment:

- Accuracy: 90.13%
- Macro Precision: 66.20%
- Macro Recall: 60.77%
- Macro F1: 60.47%

Aspect category:

- Accuracy: 62.24%
- Macro Precision: 17.99%
- Macro Recall: 19.62%
- Macro F1: 17.37%

### Baseline vs XLM-R

Sentiment:

Baseline Macro-F1: 58.58%

XLM-R Macro-F1: 60.47%

Category:

Baseline Macro-F1: 16.94%

XLM-R Macro-F1: 17.37%

The transformer model improves both tasks compared with the classical TF-IDF + Logistic Regression baseline.

---

## 7. Multilingual Evaluation

The model is evaluated separately for English, Russian, and German.

### Sentiment Results by Language

English:

- Accuracy: 90.03%
- Macro-F1: 59.54%

Russian:

- Accuracy: 90.16%
- Macro-F1: 60.93%

German:

- Accuracy: 90.17%
- Macro-F1: 60.89%

The results show similar sentiment classification accuracy across the three languages.

The main classification limitation is the neutral sentiment class.

The test confusion matrix shows:

Actual Negative:
- Correctly classified: 1,377
- Predicted Neutral: 17
- Predicted Positive: 264

Actual Neutral:
- Predicted Negative: 142
- Correctly classified: 7
- Predicted Positive: 153

Actual Positive:
- Predicted Negative: 170
- Predicted Neutral: 6
- Correctly classified: 5,475

The neutral class has substantially lower recall because of class imbalance and difficulty separating neutral statements from positive and negative opinions.

---

## 8. Sentence Embeddings

Notebook:

05_embeddings_and_retrieval.ipynb

For semantic retrieval, the project uses:

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Embedding dimension:

384

The model converts reviews into numerical vectors representing their semantic meaning.

Architecture:

Review
  |
  v
Sentence Transformer
  |
  v
384-dimensional embedding
  |
  v
FAISS

The multilingual embedding model allows queries in one language to retrieve semantically related reviews written in another language.

---

## 9. FAISS Semantic Retrieval

FAISS is used as the vector similarity search engine.

The index uses normalized embeddings and inner-product search.

Because the embeddings are normalized, the inner product corresponds to cosine similarity.

The retrieval pipeline is:

Query
  |
  v
Query Embedding
  |
  v
FAISS
  |
  v
Top-K Similar Reviews

The FAISS index is built only from training reviews to prevent test-set leakage.

The index contains 10,355 training reviews.

---

## 10. Similarity

Similarity is used to evaluate the retrieval stage of the RAG system.

For RAG retrieval:

Similarity measures the semantic similarity between:

User Query <-> Retrieved Review

Example:

Query:

"The hotel room was very clean"

Retrieved review:

"The rooms were clean and the hotel staff was friendly."

A high similarity score means that the retrieved review is semantically close to the query.

The project records:

- Top-1 similarity;
- Mean Top-K similarity.

Top-1 similarity represents the similarity of the most relevant retrieved review.

Mean Top-K similarity represents the average similarity of all retrieved reviews.

Important:

Retrieval similarity evaluates the retrieval stage.

It does not directly evaluate the quality of the generated LLM answer.

---

## 11. RAG Pipeline

Notebook:

06_rag_evaluation.ipynb

The Retrieval-Augmented Generation pipeline combines semantic retrieval with a local LLM.

Pipeline:

User Query
    |
    v
Sentence Transformer
    |
    v
Query Embedding
    |
    v
FAISS
    |
    v
Top-K Reviews
    |
    v
Context Construction
    |
    v
Qwen
    |
    v
Structured Answer

The retrieved reviews are inserted into the LLM prompt as context.

The LLM is instructed to generate a structured JSON response containing information such as:

- Main finding
- Common opinion
- Negative signals
- Key aspects
- Conclusion

---

## 12. Local LLM

The project uses a locally running Qwen model through Ollama.

Example model:

qwen3:8b

The LLM runs locally and does not require an external OpenAI API for generation.

The RAG generation process is:

Query
  |
  v
Retrieved Reviews
  |
  v
Context
  |
  v
Qwen
  |
  v
Answer

The project therefore demonstrates a fully local RAG pipeline.

---

## 13. LLM-only vs RAG

Notebook:

07_rag_llm_evaluation.ipynb

The final experiment compares two systems.

### LLM-only

The model receives only the user query.

Query
  |
  v
Qwen
  |
  v
Answer

The model does not have access to the review database.

### RAG

The model receives the query together with retrieved customer reviews.

Query
  |
  v
Embedding
  |
  v
FAISS
  |
  v
Retrieved Reviews
  |
  v
Context
  |
  v
Qwen
  |
  v
Answer

This experiment is designed to compare a general LLM response with an answer grounded in retrieved review evidence.

---

## 14. Similarity Evaluation: LLM vs RAG

The project distinguishes between two types of similarity.

### LLM-only Similarity

For the LLM-only system, semantic similarity is calculated between:

Query <-> Generated Answer

This measures how semantically related the generated answer is to the original query.

This is an answer-level semantic similarity measure.

### RAG Retrieval Similarity

For the RAG system, retrieval similarity is calculated between:

Query <-> Retrieved Reviews

This measures how relevant the retrieved context is to the query.

These measurements evaluate different stages of the system and should not be interpreted as the same metric.

Summary:

LLM-only:

Query -> Qwen -> Answer

Similarity:

Query <-> Answer


RAG:

Query -> FAISS -> Reviews

Similarity:

Query <-> Retrieved Reviews

---

## 15. Error Analysis

The project includes detailed error analysis for the transformer model.

The analysis includes:

- sentiment errors;
- sentiment confusion pairs;
- category errors;
- category confusion pairs;
- errors by language;
- errors by domain;
- neutral-class analysis.

The main observed sentiment problem is the low recall of the neutral class.

This indicates that further work could focus on:

- class balancing;
- weighted loss;
- focal loss;
- better aspect representations;
- additional training data;
- contrastive learning.

---

## 16. Retrieval Experiments

Several multilingual retrieval experiments are performed.

Examples include queries related to:

- hotel room cleanliness;
- laptop battery;
- restaurant service.

The retrieval system is tested using:

- English queries;
- Russian queries;
- German queries.

Cross-lingual retrieval is supported.

For example, an English query can retrieve semantically related German or Russian reviews.

This demonstrates the multilingual capabilities of the selected sentence-transformer model.

---

## 17. Project Structure

ReviewLens-AI/
|
├── data/
│   ├── raw/
│   │   ├── mams/
│   │   └── m_absa/
│   │
│   └── processed/
│       ├── m_absa_selected.csv
│       └── reviews.csv
|
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_absa_baseline.ipynb
│   ├── 03_transformer_training.ipynb
│   ├── 04_multilingual_evaluation.ipynb
│   ├── 05_embeddings_and_retrieval.ipynb
│   ├── 06_rag_evaluation.ipynb
│   └── 07_rag_llm_evaluation.ipynb
|
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── absa_model.py
│   │   └── embeddings.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   └── generator.py
│   │
│   └── pipeline.py
|
├── models/
│   ├── absa/
│   │   └── xlmr_multitask/
│   └── embeddings/
|
├── vector_store/
│   ├── index.faiss
│   └── metadata.pkl
|
├── app/
│   └── streamlit_app.py
|
├── tests/
│   ├── test_data.py
│   ├── test_absa.py
│   ├── test_retrieval.py
│   └── test_pipeline.py
|
├── configs/
│   └── config.yaml
|
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md

---

## 18. Installation

Clone the repository:

git clone <repository-url>

Move to the project directory:

cd ReviewLens-AI

Create a virtual environment:

python -m venv .venv

Activate the environment on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

---

## 19. Ollama Setup

Install Ollama.

Pull the required Qwen model:

ollama pull qwen3:8b

Check installed models:

ollama list

The notebooks communicate with the local Ollama instance for LLM generation.

---

## 20. Running the Project

The recommended execution order is:

1. 01_data_exploration.ipynb
2. 02_absa_baseline.ipynb
3. 03_transformer_training.ipynb
4. 04_multilingual_evaluation.ipynb
5. 05_embeddings_and_retrieval.ipynb
6. 06_rag_evaluation.ipynb
7. 07_rag_llm_evaluation.ipynb

The notebooks are designed as a progressive pipeline.

The output of earlier notebooks is reused by subsequent notebooks.

---

## 21. Technologies

Programming:

- Python
- Pandas
- NumPy

Machine Learning:

- Scikit-learn
- PyTorch

NLP:

- Hugging Face Transformers
- Hugging Face Datasets
- XLM-RoBERTa
- Sentence Transformers

Vector Search:

- FAISS

LLM:

- Ollama
- Qwen

Development:

- Jupyter Notebook
- Git
- GitHub

---

## 22. Hardware

The project was developed and tested using an NVIDIA RTX 4050 Laptop GPU.

GPU acceleration is used for:

- XLM-R training;
- XLM-R inference;
- Sentence Transformer embeddings.

FAISS uses CPU-based vector search.

The architecture is:

RTX 4050
    |
    +-- XLM-R
    |
    +-- Sentence Transformer
    |
    v
384-dimensional embeddings
    |
    v
FAISS CPU
    |
    v
Retrieved Reviews
    |
    v
Local Qwen via Ollama

---

## 23. Limitations

The current system has several limitations.

1. The dataset is imbalanced toward positive sentiment.

2. The neutral sentiment class has low recall.

3. Aspect category classification has relatively low Macro-F1 due to the large number of categories.

4. Retrieval similarity measures semantic relevance but does not directly measure factual correctness of the generated answer.

5. LLM answer similarity measures semantic correspondence between a query and answer but does not directly measure factual correctness or grounding.

6. The current RAG evaluation can be improved with dedicated retrieval relevance and answer-grounding metrics.

7. The system currently uses a limited selection of domains and languages from the full M-ABSA dataset.

---

## 24. Future Improvements

Potential improvements include:

- Class-weighted loss
- Focal loss
- Better handling of the neutral class
- Contrastive learning for sentence embeddings
- Hybrid BM25 + vector retrieval
- Cross-encoder reranking
- Retrieval precision and recall evaluation
- LLM-as-a-Judge evaluation
- Faithfulness evaluation
- Answer grounding metrics
- Larger multilingual models
- Production API
- Streamlit dashboard
- Automated evaluation pipeline
- Docker deployment

---

## 25. Key NLP Concepts Demonstrated

The project demonstrates practical knowledge of:

- Natural Language Processing
- Aspect-Based Sentiment Analysis
- Multilingual NLP
- Multitask Learning
- Transformer Fine-Tuning
- XLM-RoBERTa
- Sentence Embeddings
- Semantic Search
- Vector Search
- FAISS
- Retrieval-Augmented Generation
- Local LLMs
- Prompt Engineering
- LLM Evaluation
- Retrieval Evaluation
- Error Analysis
- Data Leakage Prevention
- Class Imbalance Analysis

---

## 26. Project Summary

ReviewLens AI demonstrates an end-to-end multilingual NLP pipeline starting from raw customer reviews and ending with an evidence-based LLM response.

The project combines:

Data Processing
    ->
Classical NLP Baseline
    ->
Multilingual Transformer
    ->
Multilingual Embeddings
    ->
FAISS Retrieval
    ->
RAG
    ->
Local Qwen LLM
    ->
Evaluation

The main objective is not only sentiment classification, but building a complete review intelligence system capable of retrieving relevant customer feedback and using it as evidence for LLM-generated answers.

---

## Author

ReviewLens AI

Multilingual Aspect-Based Sentiment Analysis & Review Intelligence