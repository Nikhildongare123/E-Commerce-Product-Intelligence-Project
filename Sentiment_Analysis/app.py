import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter

# Try to import plotly, fallback to matplotlib if not available
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Page configuration
st.set_page_config(
    page_title="TF-IDF Vectorizer Analyzer",
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

# Load the TF-IDF vectorizer with comprehensive error handling
@st.cache_resource
def load_vectorizer():
    try:
        with open('tfidf.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return vectorizer
    except FileNotFoundError:
        st.error("❌ **File Not Found**\n\n`tfidf.pkl` not found in the current directory. Please ensure the file exists.")
        return None
    except pickle.UnpicklingError:
        st.error("❌ **Corrupted File**\n\nThe file appears to be corrupted or not a valid pickle file.")
        return None
    except ModuleNotFoundError as e:
        st.error(f"❌ **Missing Dependency**\n\nRequired module not found: `{e.name}`\n\nTry installing it with: `pip install {e.name}`")
        return None
    except AttributeError as e:
        st.error(f"❌ **Attribute Error**\n\n{str(e)}\n\nThe pickle file may have been created with a different version of scikit-learn.")
        return None
    except Exception as e:
        st.error(f"❌ **Unexpected Error**\n\n{str(e)}")
        return None

# Helper function to safely extract attributes
def safe_get_attr(obj, attr, default="N/A"):
    try:
        value = getattr(obj, attr, default)
        if callable(value):
            return "Callable"
        return value
    except:
        return default

# Helper to get vocabulary
def get_vocabulary(v):
    try:
        if hasattr(v, 'vocabulary_'):
            return v.vocabulary_
        elif hasattr(v, 'get_feature_names_out'):
            try:
                features = v.get_feature_names_out()
                return {feature: idx for idx, feature in enumerate(features)}
            except:
                pass
        elif hasattr(v, '_tfidf') and hasattr(v._tfidf, 'idf_'):
            idf_len = len(v._tfidf.idf_)
            return {f"term_{i}": i for i in range(idf_len)}
        return {}
    except:
        return {}

# Helper to get IDF values
def get_idf_values(v):
    try:
        if hasattr(v, '_tfidf') and hasattr(v._tfidf, 'idf_'):
            return v._tfidf.idf_
        elif hasattr(v, 'idf_'):
            return v.idf_
        return None
    except:
        return None

vectorizer = load_vectorizer()

if vectorizer is None:
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Get vocabulary
    vocab = get_vocabulary(vectorizer)
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
        "Analyzer": safe_get_attr(vectorizer, 'analyzer'),
        "Max DF": safe_get_attr(vectorizer, 'max_df'),
        "Min DF": safe_get_attr(vectorizer, 'min_df'),
        "Max Features": safe_get_attr(vectorizer, 'max_features'),
        "NGram Range": safe_get_attr(vectorizer, 'ngram_range'),
        "Stop Words": "Enabled" if vectorizer.stop_words is not None else "Disabled",
        "Use IDF": safe_get_attr(vectorizer, 'use_idf'),
        "Smooth IDF": safe_get_attr(vectorizer, 'smooth_idf'),
        "Sublinear TF": safe_get_attr(vectorizer, 'sublinear_tf'),
    }
    
    for key, value in params.items():
        st.markdown(f"**{key}:** `{value}`")
    
    # Sklearn version
    if hasattr(vectorizer, '_sklearn_version'):
        st.markdown(f"**Sklearn Version:** `{vectorizer._sklearn_version}`")
    
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
        
        if vocab:
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
        else:
            st.warning("No vocabulary found in the vectorizer.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📝 Word Statistics")
        
        if vocab:
            words = list(vocab.keys())
            avg_len = np.mean([len(w) for w in words])
            max_len = max([len(w) for w in words])
            min_len = min([len(w) for w in words])
            
            st.metric("Average Word Length", f"{avg_len:.2f}")
            st.metric("Longest Word", f"{max_len} chars")
            st.metric("Shortest Word", f"{min_len} chars")
            
            st.markdown("---")
            st.markdown("#### 🔤 Longest Terms")
            longest_terms = sorted(vocab.items(), key=lambda x: len(x[0]), reverse=True)[:10]
            for word, idx in longest_terms:
                st.markdown(f"`{word}` ({len(word)} chars)")
        else:
            st.info("No vocabulary data available.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 2: Word Weights ---
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📊 IDF Weights Distribution")
    
    idf_values = get_idf_values(vectorizer)
    
    if idf_values is not None and len(idf_values) > 0:
        if vocab:
            idf_df = pd.DataFrame({
                "Word": list(vocab.keys())[:len(idf_values)],
                "IDF": idf_values
            })
        else:
            idf_df = pd.DataFrame({
                "Word": [f"term_{i}" for i in range(len(idf_values))],
                "IDF": idf_values
            })
        
        idf_df = idf_df.sort_values("IDF", ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📈 IDF Distribution")
            
            # Use matplotlib for histogram
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(idf_df["IDF"], bins=30, color='#4a90d9', edgecolor='white', alpha=0.8)
            ax.set_xlabel("IDF Score")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of IDF Values")
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('white')
            st.pyplot(fig)
            plt.close()
        
        with col2:
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
    
    if vocab:
        search_col1, search_col2 = st.columns([3, 1])
        
        with search_col1:
            search_term = st.text_input("Search for a term:", placeholder="e.g., 'quality', 'price', 'product'")
        
        with search_col2:
            search_type = st.selectbox("Search type:", ["Contains", "Starts with", "Ends with", "Exact match", "Regex"])
        
        if search_term:
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
                elif search_type == "Regex":
                    try:
                        if re.search(search_term, word, re.IGNORECASE):
                            filtered_vocab[word] = idx
                    except:
                        st.warning("Invalid regex pattern.")
                        break
            
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
    else:
        st.warning("No vocabulary available for searching.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 4: Statistics ---
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Dataset Statistics")
        
        stats = {
            "Total Vocabulary Size": len(vocab) if vocab else 0,
            "Analyzer Type": safe_get_attr(vectorizer, 'analyzer'),
            "NGram Range": safe_get_attr(vectorizer, 'ngram_range'),
            "Stop Words": "Custom" if vectorizer.stop_words is not None else "None",
            "Lowercase": safe_get_attr(vectorizer, 'lowercase'),
            "Binary": safe_get_attr(vectorizer, 'binary'),
            "Norm": safe_get_attr(vectorizer, 'norm'),
        }
        
        for key, value in stats.items():
            st.markdown(f"**{key}:** `{value}`")
        
        if hasattr(vectorizer, 'n_features_in_'):
            st.markdown(f"**Features In:** `{vectorizer.n_features_in_}`")
        
        if hasattr(vectorizer, 'fixed_vocabulary_'):
            st.markdown(f"**Fixed Vocabulary:** `{vectorizer.fixed_vocabulary_}`")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🏷️ Sample Terms")
        
        if vocab:
            sample_terms = list(vocab.keys())[:50]
            sample_terms_html = "".join([f'<span class="word-item">{term}</span>' for term in sample_terms])
            st.markdown(f'<div style="padding: 0.5rem 0;">{sample_terms_html}</div>', unsafe_allow_html=True)
            st.caption("Showing first 50 terms from the vocabulary")
        else:
            st.info("No vocabulary available.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if vocab:
        st.markdown('<div class="card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown("#### 📏 Term Length Distribution")
        
        lengths = [len(word) for word in vocab.keys()]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(lengths, bins=20, color='#2ecc71', edgecolor='white', alpha=0.8)
        ax.set_xlabel("Number of Characters")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Term Lengths")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
        plt.close()
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- Export Section ---
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if vocab:
        export_df = pd.DataFrame(list(vocab.items()), columns=["Word", "Index"])
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Vocabulary as CSV",
            data=csv,
            file_name="tfidf_vocabulary.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d; font-size: 0.85rem;'>"
    "🔍 Built with Streamlit • TF-IDF Vectorizer Analyzer<br>"
    "Explore the structure and weights of your text features"
    "</div>",
    unsafe_allow_html=True
)
