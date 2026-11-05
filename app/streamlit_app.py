import ast
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from torch import nn

BASE_DIR = Path(__file__).resolve().parent.parent
LOGO_PATH = BASE_DIR / "assets" / "logo.png"

st.set_page_config(
    page_title="ReviewLens AI",
    page_icon=str(LOGO_PATH),
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models' / 'absa' / 'xlmr_multitask'
DATA_PATH = ROOT / 'data' / 'processed' / 'm_absa_selected.csv'
INDEX_PATH = ROOT / 'vector_store' / 'index.faiss'
META_PATH = ROOT / 'vector_store' / 'metadata.pkl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

with st.sidebar:
    st.image(str(LOGO_PATH), width=110)
    st.markdown("## ReviewLens AI")
    st.caption("AI Review Intelligence")
    st.divider()

st.set_page_config(page_title='ReviewLens AI', page_icon='🔎', layout='wide')
st.markdown('''<style>
.block-container{padding-top:2rem;max-width:1180px}
.hero{padding:22px 26px;border-radius:18px;background:linear-gradient(110deg,#253d9e,#7a45bb);color:white;margin-bottom:20px}
.hero h1{margin:0;color:white}.hero p{margin:8px 0 0;color:#edf0ff}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🔎 ReviewLens AI</h1><p>Multilingual aspect-based sentiment analysis · Semantic retrieval · Evidence-grounded AI</p></div>', unsafe_allow_html=True)

class MultiTaskXLMR(nn.Module):
    def __init__(self, model_name, num_sentiment_classes, num_category_classes, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.sentiment_classifier = nn.Linear(hidden_size, num_sentiment_classes)
        self.category_classifier = nn.Linear(hidden_size, num_category_classes)
        self.sentiment_loss_fn = nn.CrossEntropyLoss()
        self.category_loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, sentiment_labels=None, category_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])
        sentiment_logits = self.sentiment_classifier(pooled)
        category_logits = self.category_classifier(pooled)
        loss = None
        if sentiment_labels is not None and category_labels is not None:
            loss = self.sentiment_loss_fn(sentiment_logits, sentiment_labels) + self.category_loss_fn(category_logits, category_labels)
        return {'loss': loss, 'sentiment_logits': sentiment_logits, 'category_logits': category_logits}

@st.cache_resource(show_spinner='Loading trained XLM-RoBERTa weights...')
def load_absa():
    if not (MODEL_DIR / 'model.pt').exists():
        raise FileNotFoundError(f'Weights not found: {MODEL_DIR / "model.pt"}')
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATA_PATH}')
    df = pd.read_csv(DATA_PATH, keep_default_na=False)
    sentiments = sorted(df['sentiment'].unique().tolist())
    categories = sorted(df['category'].unique().tolist())
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = MultiTaskXLMR('xlm-roberta-base', len(sentiments), len(categories))
    state = torch.load(MODEL_DIR / 'model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return model, tokenizer, sentiments, categories

@st.cache_resource(show_spinner='Loading multilingual embeddings and FAISS...')
def load_search():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError('FAISS index.faiss or metadata.pkl is missing')
    embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=DEVICE)
    index = faiss.read_index(str(INDEX_PATH))
    metadata = pd.read_pickle(META_PATH).reset_index(drop=True)
    if index.ntotal != len(metadata):
        raise ValueError(f'FAISS has {index.ntotal} vectors but metadata has {len(metadata)} rows')
    return embedder, index, metadata

def predict(text, aspect):
    model, tokenizer, sentiments, categories = load_absa()
    tokens = tokenizer(text, aspect, truncation=True, max_length=128, return_tensors='pt')
    tokens = {k: v.to(DEVICE) for k, v in tokens.items() if k in ('input_ids', 'attention_mask')}
    with torch.inference_mode():
        output = model(**tokens)
        sentiment_probs = torch.softmax(output['sentiment_logits'], dim=-1)[0].cpu().numpy()
        category_probs = torch.softmax(output['category_logits'], dim=-1)[0].cpu().numpy()
    return sentiments[int(sentiment_probs.argmax())], categories[int(category_probs.argmax())], dict(zip(sentiments, sentiment_probs.tolist())), float(category_probs.max())

def retrieve(query, top_k=5, domain='All', language='All'):
    embedder, index, metadata = load_search()
    vec = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
    # Search the whole index to avoid filtering away valid results after a small initial top-k.
    k = index.ntotal if (domain != 'All' or language != 'All') else min(top_k, index.ntotal)
    scores, ids = index.search(vec, k)
    valid = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
    result = metadata.iloc[[i for i, _ in valid]].copy()
    result['score'] = [s for _, s in valid]
    if domain != 'All':
        result = result[result['domain'] == domain]
    if language != 'All':
        result = result[result['language'] == language]
    return result.head(top_k).reset_index(drop=True)

def answer_question(query, results):
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError('Install the Ollama Python package: pip install ollama') from exc
    context = '\n'.join(f'[{i+1}] ({row.language}, {row.domain}; similarity={row.score:.3f}) {row.text}' for i, row in enumerate(results.itertuples()))
    prompt = f'''You are ReviewLens AI, a review analysis assistant. Answer the question using ONLY the provided customer reviews. Do not invent evidence. If the evidence is insufficient, say so. Cite reviews by [number]. Respond in the same language as the question.\n\nQUESTION:\n{query}\n\nRETRIEVED REVIEWS:\n{context}\n\nWrite a concise summary, positive and negative signals, and evidence citations.'''
    response = ollama.chat(model='qwen3:8b', messages=[{'role': 'user', 'content': prompt}], options={'temperature': 0.2})
    return response.message.content

with st.sidebar:
    st.header('⚙️ System')
    st.caption(f'Device: {DEVICE.upper()}')
    st.caption('ABSA: XLM-RoBERTa · Qwen3:8b · FAISS')
    st.info('ABSA accepts a review + a specified aspect. Automatic aspect extraction is not implemented.')
    st.caption('Run from project root: streamlit run app/streamlit_app.py')

tab1, tab2, tab3 = st.tabs(['🧠 Review Analyzer', '🔍 Semantic Search', '✨ AI Assistant'])

with tab1:
    st.subheader('Analyze a review')
    review = st.text_area('Customer review', 'The food was delicious, but the service was slow.', height=115)
    aspect = st.text_input('Aspect to analyze', 'service')
    if st.button('Analyze sentiment', type='primary'):
        if not review.strip() or not aspect.strip():
            st.warning('Enter both a review and an aspect.')
        else:
            try:
                with st.spinner('Running your trained model...'):
                    sentiment, category, probabilities, cat_conf = predict(review.strip(), aspect.strip())
                c1, c2, c3 = st.columns(3)
                c1.metric('Sentiment', sentiment.title())
                c2.metric('Aspect category', category)
                c3.metric('Category confidence', f'{cat_conf:.1%}')
                st.write('**Sentiment probabilities**')
                st.bar_chart(pd.DataFrame({'Probability': probabilities}))
                st.caption('Probabilities are model softmax outputs and may be poorly calibrated.')
            except Exception as exc:
                st.error(f'Unable to load or run ABSA model: {exc}')

with tab2:
    st.subheader('Find semantically similar reviews')
    search_query = st.text_input('Search query', 'The hotel room was clean and comfortable', key='search_query')
    c1, c2, c3 = st.columns(3)
    domain = c1.selectbox('Domain', ['All', 'restaurant', 'hotel', 'laptop'], key='search_domain')
    language = c2.selectbox('Language', ['All', 'en', 'ru', 'de'], key='search_lang')
    top_k = c3.slider('Number of reviews', 1, 10, 5, key='search_k')
    if st.button('Search reviews', type='primary'):
        if not search_query.strip():
            st.warning('Enter a query.')
        else:
            try:
                with st.spinner('Searching FAISS index...'):
                    matches = retrieve(search_query, top_k, domain, language)
                if matches.empty:
                    st.info('No matching reviews found.')
                else:
                    cols = [x for x in ['text', 'language', 'domain', 'score'] if x in matches.columns]
                    st.dataframe(matches[cols], use_container_width=True, hide_index=True)
                    st.caption('Similarity scores are cosine similarity for normalized vectors, not model accuracy.')
            except Exception as exc:
                st.error(f'Search failed: {exc}')

with tab3:
    st.subheader('Ask questions about customer feedback')
    question = st.text_area('Your question', 'What do customers say about hotel room cleanliness?', height=90)
    c1, c2, c3 = st.columns(3)
    rag_domain = c1.selectbox('Domain', ['All', 'restaurant', 'hotel', 'laptop'], key='rag_domain')
    rag_language = c2.selectbox('Language', ['All', 'en', 'ru', 'de'], key='rag_lang')
    rag_k = c3.slider('Evidence reviews', 1, 10, 5)
    if st.button('Generate evidence-based answer', type='primary'):
        if not question.strip():
            st.warning('Enter a question.')
        else:
            try:
                with st.spinner('Retrieving evidence...'):
                    matches = retrieve(question.strip(), rag_k, rag_domain, rag_language)
                if matches.empty:
                    st.info('No evidence found for the selected filters.')
                else:
                    with st.spinner('Asking local Qwen3...'):
                        answer = answer_question(question.strip(), matches)
                    st.markdown(answer)
                    with st.expander('Show retrieved evidence'):
                        st.dataframe(matches[['text', 'language', 'domain', 'score']], use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f'RAG generation failed: {exc}')
