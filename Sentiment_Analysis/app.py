import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import os
from collections import Counter
import glob

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
    
    /* Progress bar for IDF */
    .idf-bar-container {
        background: #e9ecef;
        border-radius: 4px;
        height: 6px;
        margin: 4px 0;
        overflow: hidden;
    }
    
    .idf-bar {
        background: #4a90d9;
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s;
    }
    
    .file-item {
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 6px;
        background: #f8f9fa;
        border-left: 3px solid #4a90d9;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Function to find the tfidf.pkl file
def find_tfidf_file():
    """Search for tfidf.pkl in various locations"""
    search_paths = [
        '.',  # Current directory
        '..',  # Parent directory
        '../..',  # Two levels up
        'Sentiment_Analysis',  # Sentiment_Analysis folder
        'models',  # Models folder
        'data',  # Data folder
        'outputs',  # Outputs folder
        'artifacts',  # Artifacts folder
    ]
    
    found_files = []
    for path in search_paths:
        try:
            pattern = os.path.join(path, '*.pkl')
            for file in glob.glob(pattern):
                if 'tfidf' in file.lower():
                    found_files.append(file)
        except:
            continue
    
    # Also check for tfidf.pkl directly
    for path in search_paths:
        try:
            full_path = os.path.join(path, 'tfidf.pkl')
            if os.path.exists(full_path):
                if full_path not in found_files:
                    found_files.append(full_path)
        except:
            continue
    
    return found_files

# Load the TF-IDF vectorizer with file discovery
@st.cache_resource
def load_vectorizer(file_path=None):
    """Load vectorizer from specified path or auto-discover"""
    
    # If file_path is provided, try to load it
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                vectorizer = pickle.load(f)
            return vectorizer, file_path
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None, None
    
    # Auto-discover the file
    found_files = find_tfidf_file()
    
    if found_files:
        # Try loading the first found file
        for file_path in found_files:
            try:
                with open(file_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                return vectorizer, file_path
            except Exception as e:
                continue
    
    return None, None

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

# --- File Upload Section ---
st.markdown("<h1 style='margin-bottom: 0.2rem;'>📊 TF-IDF Vectorizer Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #6c757d; font-size: 1.1rem; margin-bottom: 2rem;'>Explore the vocabulary, weights, and structure of your TF-IDF model</p>", unsafe_allow_html=True)

# Initialize session state for vectorizer and file path
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'file_path' not in st.session_state:
    st.session_state.file_path = None
if 'vocab' not in st.session_state:
    st.session_state.vocab = None

# File selection/upload section
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📂 Load TF-IDF File")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Option 1: Upload file
        uploaded_file = st.file_uploader(
            "Upload tfidf.pkl file",
            type=['pkl'],
            help="Upload your tfidf.pkl file directly"
        )
        
        if uploaded_file is not None:
            try:
                vectorizer = pickle.load(uploaded_file)
                st.session_state.vectorizer = vectorizer
                st.session_state.file_path = "Uploaded file"
                st.session_state.vocab = get_vocabulary(vectorizer)
                st.success("✅ File loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading uploaded file: {str(e)}")
    
    with col2:
        # Option 2: Auto-discover
        if st.button("🔍 Auto-Discover tfidf.pkl", use_container_width=True):
            vectorizer, file_path = load_vectorizer()
            if vectorizer is not None:
                st.session_state.vectorizer = vectorizer
                st.session_state.file_path = file_path
                st.session_state.vocab = get_vocabulary(vectorizer)
                st.success(f"✅ Found and loaded: `{file_path}`")
                st.rerun()
            else:
                st.error("❌ Could not find tfidf.pkl in common locations.")
    
    # Show found files if any
    found_files = find_tfidf_file()
    if found_files and not st.session_state.vectorizer:
        st.markdown("**📁 Found these pickle files in your project:**")
        for file in found_files[:5]:
            st.markdown(f'<div class="file-item">📄 {file}</div>', unsafe_allow_html=True)
        if len(found_files) > 5:
            st.caption(f"And {len(found_files) - 5} more files...")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main App ---
if st.session_state.vectorizer is not None:
    vectorizer = st.session_state.vectorizer
    vocab = st.session_state.vocab
    file_path = st.session_state.file_path
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"### 📂 Loaded File")
        st.info(f"`{file_path}`")
        
        st.markdown("### ⚙️ Configuration")
        
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
        
        if hasattr(vectorizer, '_sklearn_version'):
            st.markdown(f"**Sklearn Version:** `{vectorizer._sklearn_version}`")
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #6c757d; padding: 1rem 0;">
            🔍 TF-IDF (Term Frequency-Inverse Document Frequency)<br>
            A numerical statistic that reflects the importance of a word in a document.
        </div>
        """, unsafe_allow_html=True)
    
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
        st.markdown("#### 📊 IDF Weights")
        
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
            
            # Display IDF statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min IDF", f"{idf_df['IDF'].min():.3f}")
            with col2:
                st.metric("Max IDF", f"{idf_df['IDF'].max():.3f}")
            with col3:
                st.metric("Mean IDF", f"{idf_df['IDF'].mean():.3f}")
            
            st.markdown("---")
            
            # Show top terms with bar visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⬆️ Most Important (High IDF)")
                top_idf = idf_df.nlargest(20, "IDF")
                max_idf = top_idf["IDF"].max()
                
                for _, row in top_idf.iterrows():
                    pct = (row["IDF"] / max_idf) * 100
                    st.markdown(f"""
                    <div style="margin-bottom: 4px;">
                        <span style="font-size: 0.85rem;">`{row['Word']}`</span>
                        <span style="float: right; font-size: 0.8rem; color: #6c757d;">{row['IDF']:.3f}</span>
                        <div class="idf-bar-container">
                            <div class="idf-bar" style="width: {pct}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### ⬇️ Least Important (Low IDF)")
                bottom_idf = idf_df.nsmallest(20, "IDF")
                max_idf = bottom_idf["IDF"].max() if bottom_idf["IDF"].max() > 0 else 1
                
                for _, row in bottom_idf.iterrows():
                    pct = (row["IDF"] / max_idf) * 100 if max_idf > 0 else 0
                    st.markdown(f"""
                    <div style="margin-bottom: 4px;">
                        <span style="font-size: 0.85rem;">`{row['Word']}`</span>
                        <span style="float: right; font-size: 0.8rem; color: #6c757d;">{row['IDF']:.3f}</span>
                        <div class="idf-bar-container">
                            <div class="idf-bar" style="width: {pct}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
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
        
        # Term length distribution using Streamlit's built-in components
        if vocab:
            st.markdown('<div class="card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown("#### 📏 Term Length Distribution")
            
            # Calculate length distribution
            lengths = [len(word) for word in vocab.keys()]
            length_counts = Counter(lengths)
            length_df = pd.DataFrame([
                {"Length": k, "Count": v} for k, v in sorted(length_counts.items())
            ])
            
            if not length_df.empty:
                st.bar_chart(length_df.set_index("Length"), height=300)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min Length", min(lengths))
                with col2:
                    st.metric("Max Length", max(lengths))
                with col3:
                    st.metric("Average Length", f"{np.mean(lengths):.1f}")
            
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

else:
    # Show instructions when no file is loaded
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📖 How to Use This App")
    
    st.markdown("""
    **This app analyzes a TF-IDF vectorizer pickle file. Here are the ways to load your file:**
    
    1. **📤 Upload** - Click "Browse files" and select your `tfidf.pkl` file
    2. **🔍 Auto-Discover** - Click the "Auto-Discover" button to search for `tfidf.pkl` in common locations
    3. **📁 Manual Placement** - Place the file in the same directory as this app and use auto-discover
    
    **After loading, you'll be able to:**
    - Explore the complete vocabulary
    - View IDF weights and importance scores
    - Search for specific terms
    - See statistics about the vocabulary
    - Export the vocabulary as CSV
    
    **Common file locations in this project:**
    - Root directory: `/tfidf.pkl`
    - Sentiment Analysis: `/Sentiment_Analysis/tfidf.pkl`
    - Models folder: `/models/tfidf.pkl`
    - Data folder: `/data/tfidf.pkl`
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
