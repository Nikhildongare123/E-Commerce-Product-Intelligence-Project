import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="TF-IDF Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a1a2e;
        font-weight: 600 !important;
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        border-left: 4px solid #4a90d9;
        margin-bottom: 0.8rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Word cloud styling */
    .word-item {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        margin: 0.2rem;
        border-radius: 20px;
        background: #e9ecef;
        color: #1a1a2e;
        font-size: 0.9rem;
        transition: all 0.2s;
    }
    
    .word-item:hover {
        background: #4a90d9;
        color: white;
        transform: scale(1.05);
        cursor: pointer;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f1f3f5;
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        background-color: transparent;
        color: #495057;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4a90d9 !important;
        color: white !important;
    }
    
    /* DataFrame */
    .dataframe {
        border: none !important;
    }
    
    .dataframe th {
        background-color: #f1f3f5 !important;
        color: #1a1a2e !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load the TF-IDF vectorizer
@st.cache_resource
def load_vectorizer():
    try:
        with open('tfidf.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return vectorizer
    except FileNotFoundError:
        st.error("❌ `tfidf.pkl` not found. Please ensure the file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

vectorizer = load_vectorizer()

if vectorizer is None:
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Get vocabulary
    vocab = vectorizer.vocabulary_
    vocab_size = len(vocab)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Vocabulary Size</div>
        <div class="metric-value">{vocab_size:,}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display parameters
    st.markdown("---")
    st.markdown("### 📋 Parameters")
    
    params = {
        "Analyzer": vectorizer.analyzer if hasattr(vectorizer, 'analyzer') else "word",
        "Max DF": vectorizer.max_df if hasattr(vectorizer, 'max_df') else "1.0",
        "Min DF": vectorizer.min_df if hasattr(vectorizer, 'min_df') else "1",
        "Max Features": vectorizer.max_features if hasattr(vectorizer, 'max_features') else "None",
        "NGram Range": vectorizer.ngram_range if hasattr(vectorizer, 'ngram_range') else "(1, 1)",
        "Stop Words": "Enabled" if vectorizer.stop_words is not None else "Disabled",
        "Use IDF": vectorizer.use_idf if hasattr(vectorizer, 'use_idf') else True,
        "Smooth IDF": vectorizer.smooth_idf if hasattr(vectorizer, 'smooth_idf') else True,
        "Sublinear TF": vectorizer.sublinear_tf if hasattr(vectorizer, 'sublinear_tf') else False,
    }
    
    for key, value in params.items():
        st.markdown(f"**{key}:** `{value}`")
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #6c757d; padding: 1rem 0;">
        🔍 TF-IDF (Term Frequency-Inverse Document Frequency)<br>
        A numerical statistic that reflects the importance of a word in a document.
    </div>
    """, unsafe_allow_html=True)

# --- Main Content ---
st.markdown("<h1 style='margin-bottom: 0.2rem;'>📊 TF-IDF Vectorizer Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #6c757d; font-size: 1.1rem; margin-bottom: 2rem;'>Explore the vocabulary, weights, and structure of your TF-IDF model</p>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📚 Vocabulary Explorer",
    "📈 Word Weights",
    "🔍 Search & Filter",
    "📊 Statistics"
])

# --- Tab 1: Vocabulary Explorer ---
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📖 Complete Vocabulary")
        
        # Convert vocab to DataFrame
        vocab_items = sorted(vocab.items(), key=lambda x: x[1])
        vocab_df = pd.DataFrame(vocab_items, columns=["Word", "Index"])
        vocab_df["Word Length"] = vocab_df["Word"].str.len()
        
        st.dataframe(
            vocab_df,
            use_container_width=True,
            height=500,
            column_config={
                "Word": st.column_config.TextColumn("Word", width="medium"),
                "Index": st.column_config.NumberColumn("Index", width="small"),
                "Word Length": st.column_config.NumberColumn("Length", width="small"),
            }
        )
        st.caption(f"Showing {len(vocab_df):,} unique terms")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📝 Word Statistics")
        
        # Basic stats
        avg_len = vocab_df["Word Length"].mean()
        max_len = vocab_df["Word Length"].max()
        min_len = vocab_df["Word Length"].min()
        
        st.metric("Average Word Length", f"{avg_len:.2f}")
        st.metric("Longest Word", f"{max_len} chars")
        st.metric("Shortest Word", f"{min_len} chars")
        
        # Top words by length
        st.markdown("---")
        st.markdown("#### 🔤 Longest Terms")
        longest_terms = vocab_df.nlargest(10, "Word Length")
        for _, row in longest_terms.iterrows():
            st.markdown(f"`{row['Word']}` ({row['Word Length']} chars)")
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 2: Word Weights ---
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📊 IDF Weights Distribution")
    
    # Check if idf_ exists
    if hasattr(vectorizer, '_tfidf') and hasattr(vectorizer._tfidf, 'idf_'):
        idf_values = vectorizer._tfidf.idf_
        
        # Create DataFrame with words and their IDF values
        idf_df = pd.DataFrame({
            "Word": list(vocab.keys()),
            "IDF": idf_values,
            "Log_IDF": np.log(idf_values)
        })
        idf_df = idf_df.sort_values("IDF", ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Histogram of IDF values
            fig = px.histogram(
                idf_df,
                x="IDF",
                nbins=30,
                title="Distribution of IDF Values",
                color_discrete_sequence=["#4a90d9"]
            )
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis_title="IDF Score",
                yaxis_title="Frequency",
                height=400,
                showlegend=False,
            )
            fig.update_xaxis(gridcolor="#f1f3f5")
            fig.update_yaxis(gridcolor="#f1f3f5")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top terms by IDF (most important)
            st.markdown("#### ⬆️ Most Important Terms (High IDF)")
            top_idf = idf_df.nlargest(15, "IDF")
            for _, row in top_idf.iterrows():
                st.markdown(f"`{row['Word']}` → {row['IDF']:.3f}")
            
            st.markdown("---")
            st.markdown("#### ⬇️ Least Important Terms (Low IDF)")
            bottom_idf = idf_df.nsmallest(15, "IDF")
            for _, row in bottom_idf.iterrows():
                st.markdown(f"`{row['Word']}` → {row['IDF']:.3f}")
    
    else:
        st.info("ℹ️ IDF weights are not available in this vectorizer.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 3: Search & Filter ---
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Search Terms")
    
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        search_term = st.text_input("Search for a term:", placeholder="e.g., 'quality', 'price', 'product'")
    
    with search_col2:
        search_type = st.selectbox("Search type:", ["Contains", "Starts with", "Ends with", "Exact match"])
    
    if search_term:
        # Filter vocabulary
        filtered_vocab = {}
        
        for word, idx in vocab.items():
            if search_type == "Contains" and search_term.lower() in word.lower():
                filtered_vocab[word] = idx
            elif search_type == "Starts with" and word.lower().startswith(search_term.lower()):
                filtered_vocab[word] = idx
            elif search_type == "Ends with" and word.lower().endswith(search_term.lower()):
                filtered_vocab[word] = idx
            elif search_type == "Exact match" and word.lower() == search_term.lower():
                filtered_vocab[word] = idx
        
        if filtered_vocab:
            filtered_df = pd.DataFrame(
                sorted(filtered_vocab.items(), key=lambda x: x[1]),
                columns=["Word", "Index"]
            )
            st.dataframe(filtered_df, use_container_width=True)
            st.caption(f"Found {len(filtered_df)} matching terms")
        else:
            st.warning("No matching terms found.")
    else:
        st.info("💡 Enter a search term above to filter the vocabulary.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 4: Statistics ---
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Dataset Statistics")
        
        stats = {
            "Total Vocabulary Size": len(vocab),
            "Unique Terms": len(vocab),
            "Analyzer Type": vectorizer.analyzer if hasattr(vectorizer, 'analyzer') else "word",
            "NGram Range": vectorizer.ngram_range if hasattr(vectorizer, 'ngram_range') else "(1, 1)",
            "Stop Words": "Custom" if vectorizer.stop_words is not None else "None",
        }
        
        for key, value in stats.items():
            st.markdown(f"**{key}:** `{value}`")
        
        # Additional info if available
        if hasattr(vectorizer, 'n_features_in_'):
            st.markdown(f"**Features In:** `{vectorizer.n_features_in_}`")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🏷️ Sample Terms")
        
        # Show first 50 terms
        sample_terms = list(vocab.keys())[:50]
        sample_terms_html = "".join([f'<span class="word-item">{term}</span>' for term in sample_terms])
        st.markdown(f'<div style="padding: 0.5rem 0;">{sample_terms_html}</div>', unsafe_allow_html=True)
        
        st.caption("Showing first 50 terms from the vocabulary")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional row for term length distribution
    st.markdown('<div class="card" style="margin-top: 1rem;">', unsafe_allow_html=True)
    st.markdown("#### 📏 Term Length Distribution")
    
    # Create term length data
    lengths = [len(word) for word in vocab.keys()]
    length_df = pd.DataFrame({"Word Length": lengths})
    
    fig = px.histogram(
        length_df,
        x="Word Length",
        nbins=20,
        title="Distribution of Term Lengths",
        color_discrete_sequence=["#2ecc71"]
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Number of Characters",
        yaxis_title="Frequency",
        height=300,
        showlegend=False,
    )
    fig.update_xaxis(gridcolor="#f1f3f5")
    fig.update_yaxis(gridcolor="#f1f3f5")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d; font-size: 0.85rem;'>"
    "🔍 Built with Streamlit • TF-IDF Vectorizer Analyzer<br>"
    "Explore the structure and weights of your text features"
    "</div>",
    unsafe_allow_html=True
)
