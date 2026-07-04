import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="Sentiment Model Analyzer",
    page_icon="🧠",
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
    
    /* Class labels */
    .class-positive {
        color: #28a745;
        font-weight: 600;
    }
    .class-negative {
        color: #dc3545;
        font-weight: 600;
    }
    .class-neutral {
        color: #ffc107;
        font-weight: 600;
    }
    
    /* Progress bars */
    .prob-bar-container {
        background: #e9ecef;
        border-radius: 6px;
        height: 8px;
        margin: 4px 0;
        overflow: hidden;
    }
    
    .prob-bar {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s;
    }
    
    .prob-bar-positive {
        background: linear-gradient(90deg, #28a745, #20c997);
    }
    
    .prob-bar-negative {
        background: linear-gradient(90deg, #dc3545, #e74c3c);
    }
    
    .prob-bar-neutral {
        background: linear-gradient(90deg, #ffc107, #fd7e14);
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
    
    .sentiment-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .sentiment-positive {
        background: #d4edda;
        color: #155724;
    }
    
    .sentiment-negative {
        background: #f8d7da;
        color: #721c24;
    }
    
    .sentiment-neutral {
        background: #fff3cd;
        color: #856404;
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

# Function to find the sentiment_model.pkl file
def find_model_file():
    """Search for sentiment_model.pkl in various locations"""
    import glob
    import os
    
    search_paths = [
        '.',  # Current directory
        '..',  # Parent directory
        '../..',  # Two levels up
        'Sentiment_Analysis',  # Sentiment_Analysis folder
        'models',  # Models folder
        'data',  # Data folder
        'outputs',  # Outputs folder
        'artifacts',  # Artifacts folder
        'model',  # Model folder
    ]
    
    found_files = []
    for path in search_paths:
        try:
            pattern = os.path.join(path, '*.pkl')
            for file in glob.glob(pattern):
                if 'sentiment' in file.lower() or 'naive' in file.lower():
                    found_files.append(file)
        except:
            continue
    
    # Also check for sentiment_model.pkl directly
    for path in search_paths:
        try:
            full_path = os.path.join(path, 'sentiment_model.pkl')
            if os.path.exists(full_path):
                if full_path not in found_files:
                    found_files.append(full_path)
        except:
            continue
    
    return found_files

# Load the sentiment model with file discovery
@st.cache_resource
def load_model(file_path=None):
    """Load model from specified path or auto-discover"""
    
    # If file_path is provided, try to load it
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            return model, file_path
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None, None
    
    # Auto-discover the file
    found_files = find_model_file()
    
    if found_files:
        for file_path in found_files:
            try:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
                return model, file_path
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

# Helper to get class labels
def get_class_labels(model):
    try:
        if hasattr(model, 'classes_'):
            return model.classes_
        return None
    except:
        return None

# Helper to get feature log probabilities
def get_feature_log_probs(model):
    try:
        if hasattr(model, 'feature_log_prob_'):
            return model.feature_log_prob_
        return None
    except:
        return None

# Helper to get feature counts
def get_feature_counts(model):
    try:
        if hasattr(model, 'feature_count_'):
            return model.feature_count_
        return None
    except:
        return None

# --- File Upload Section ---
st.markdown("<h1 style='margin-bottom: 0.2rem;'>🧠 Sentiment Model Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #6c757d; font-size: 1.1rem; margin-bottom: 2rem;'>Analyze your Multinomial Naive Bayes sentiment classification model</p>", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'file_path' not in st.session_state:
    st.session_state.file_path = None
if 'classes' not in st.session_state:
    st.session_state.classes = None

# File selection/upload section
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📂 Load Sentiment Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Option 1: Upload file
        uploaded_file = st.file_uploader(
            "Upload sentiment_model.pkl file",
            type=['pkl'],
            help="Upload your sentiment_model.pkl file directly"
        )
        
        if uploaded_file is not None:
            try:
                model = pickle.load(uploaded_file)
                st.session_state.model = model
                st.session_state.file_path = "Uploaded file"
                st.session_state.classes = get_class_labels(model)
                st.success("✅ Model loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading uploaded file: {str(e)}")
    
    with col2:
        # Option 2: Auto-discover
        if st.button("🔍 Auto-Discover Model", use_container_width=True):
            model, file_path = load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.file_path = file_path
                st.session_state.classes = get_class_labels(model)
                st.success(f"✅ Found and loaded: `{file_path}`")
                st.rerun()
            else:
                st.error("❌ Could not find sentiment_model.pkl in common locations.")
    
    # Show found files if any
    found_files = find_model_file()
    if found_files and not st.session_state.model:
        st.markdown("**📁 Found these pickle files in your project:**")
        for file in found_files[:5]:
            st.markdown(f'<div class="file-item">📄 {file}</div>', unsafe_allow_html=True)
        if len(found_files) > 5:
            st.caption(f"And {len(found_files) - 5} more files...")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main App ---
if st.session_state.model is not None:
    model = st.session_state.model
    classes = st.session_state.classes
    file_path = st.session_state.file_path
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"### 📂 Loaded File")
        st.info(f"`{file_path}`")
        
        st.markdown("### ⚙️ Model Configuration")
        
        # Display model parameters
        params = {
            "Model Type": "Multinomial Naive Bayes",
            "Alpha": safe_get_attr(model, 'alpha'),
            "Fit Prior": safe_get_attr(model, 'fit_prior'),
            "Force Alpha": safe_get_attr(model, 'force_alpha'),
            "Features In": safe_get_attr(model, 'n_features_in_'),
            "Classes": len(classes) if classes is not None else "N/A",
        }
        
        for key, value in params.items():
            st.markdown(f"**{key}:** `{value}`")
        
        if hasattr(model, '_sklearn_version'):
            st.markdown(f"**Sklearn Version:** `{model._sklearn_version}`")
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #6c757d; padding: 1rem 0;">
            🧠 Multinomial Naive Bayes<br>
            A probabilistic classifier based on Bayes' theorem<br>
            commonly used for text classification and sentiment analysis.
        </div>
        """, unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Overview",
        "📈 Feature Analysis",
        "🔍 Predict Sentiment",
        "📋 Class Statistics"
    ])
    
    # --- Tab 1: Model Overview ---
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Class Distribution")
            
            if hasattr(model, 'class_count_'):
                class_counts = model.class_count_
                class_names = classes if classes is not None else [f"Class {i}" for i in range(len(class_counts))]
                
                # Create DataFrame
                class_df = pd.DataFrame({
                    "Class": class_names,
                    "Count": class_counts,
                    "Percentage": (class_counts / class_counts.sum() * 100)
                })
                
                # Display as bar chart
                st.bar_chart(class_df.set_index("Class")["Count"], height=300)
                
                # Show table
                st.dataframe(class_df, use_container_width=True)
                
                # Show class labels with colors
                for i, name in enumerate(class_names):
                    if 'positive' in name.lower() or 'pos' in name.lower():
                        badge_class = "sentiment-positive"
                        emoji = "😊"
                    elif 'negative' in name.lower() or 'neg' in name.lower():
                        badge_class = "sentiment-negative"
                        emoji = "😞"
                    else:
                        badge_class = "sentiment-neutral"
                        emoji = "😐"
                    
                    st.markdown(f"""
                    <div style="margin: 0.3rem 0;">
                        <span class="sentiment-badge {badge_class}">{emoji} {name}</span>
                        <span style="float: right; font-weight: 600;">{class_counts[i]:.0f} samples</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Class count information not available.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Model Parameters")
            
            # Show key parameters
            param_data = {
                "Parameter": [
                    "Algorithm",
                    "Alpha (Smoothing)",
                    "Fit Prior",
                    "Force Alpha",
                    "Number of Classes",
                    "Features",
                    "Total Samples"
                ],
                "Value": [
                    "Multinomial Naive Bayes",
                    safe_get_attr(model, 'alpha'),
                    "Yes" if safe_get_attr(model, 'fit_prior') else "No",
                    "Yes" if safe_get_attr(model, 'force_alpha') else "No",
                    len(classes) if classes is not None else "N/A",
                    safe_get_attr(model, 'n_features_in_'),
                    safe_get_attr(model, 'class_count_').sum() if hasattr(model, 'class_count_') else "N/A"
                ]
            }
            
            param_df = pd.DataFrame(param_data)
            st.dataframe(param_df, use_container_width=True, hide_index=True)
            
            # Log Prior
            if hasattr(model, 'class_log_prior_'):
                st.markdown("---")
                st.markdown("#### 📐 Class Log Prior")
                for i, name in enumerate(classes if classes is not None else [f"Class {i}" for i in range(len(model.class_log_prior_))]):
                    st.metric(name, f"{model.class_log_prior_[i]:.4f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Tab 2: Feature Analysis ---
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📈 Feature Importance Analysis")
        
        feature_log_probs = get_feature_log_probs(model)
        
        if feature_log_probs is not None and classes is not None:
            # Get top features for each class
            n_top_features = st.slider("Number of top features to show:", 5, 30, 10)
            
            # For each class, get top features
            class_names = classes
            
            for i, class_name in enumerate(class_names):
                st.markdown(f"##### {class_name} - Top {n_top_features} Features")
                
                # Get log probabilities for this class
                log_probs = feature_log_probs[i]
                
                # Get indices of top features (highest log probability)
                top_indices = np.argsort(log_probs)[-n_top_features:][::-1]
                top_features = []
                
                for idx in top_indices:
                    # Try to get feature name (if we have vocabulary)
                    feature_name = f"feature_{idx}"
                    if hasattr(model, 'feature_names_in_'):
                        try:
                            feature_name = model.feature_names_in_[idx]
                        except:
                            pass
                    elif hasattr(model, 'vocabulary_'):
                        try:
                            # Reverse lookup
                            for word, word_idx in model.vocabulary_.items():
                                if word_idx == idx:
                                    feature_name = word
                                    break
                        except:
                            pass
                    
                    top_features.append({
                        "Feature": feature_name,
                        "Log Probability": log_probs[idx],
                        "Probability": np.exp(log_probs[idx])
                    })
                
                # Show as DataFrame
                feat_df = pd.DataFrame(top_features)
                feat_df["Probability"] = feat_df["Probability"].apply(lambda x: f"{x:.6f}")
                
                # Show progress bars for probabilities
                st.dataframe(feat_df, use_container_width=True)
                
                # Visualize with bars
                prob_values = [np.exp(f["Log Probability"]) for f in top_features]
                max_prob = max(prob_values) if prob_values else 1
                
                for f in top_features[:10]:
                    prob = np.exp(f["Log Probability"])
                    pct = (prob / max_prob) * 100 if max_prob > 0 else 0
                    st.markdown(f"""
                    <div style="margin-bottom: 3px;">
                        <span style="font-size: 0.85rem;">`{f['Feature']}`</span>
                        <span style="float: right; font-size: 0.8rem; color: #6c757d;">{prob:.6f}</span>
                        <div class="prob-bar-container">
                            <div class="prob-bar prob-bar-positive" style="width: {pct}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
        else:
            st.info("ℹ️ Feature log probabilities not available in this model.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Tab 3: Predict Sentiment ---
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🔍 Predict Sentiment")
        
        st.markdown("""
        Enter text below to see how the model classifies it and get probability scores for each sentiment class.
        """)
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="e.g., This product is amazing! I love it.",
            height=100
        )
        
        # Manual prediction using model internals
        def predict_manual(text, model, classes):
            """Manual prediction for demonstration"""
            try:
                # If model has predict_proba, use it
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba([text])[0]
                    prediction = model.predict([text])[0]
                    return prediction, proba
                else:
                    # Fallback - just show what we can
                    return None, None
            except:
                return None, None
        
        if text_input:
            prediction, proba = predict_manual(text_input, model, classes)
            
            if prediction is not None and proba is not None:
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    # Show prediction
                    pred_label = prediction if isinstance(prediction, str) else str(prediction)
                    
                    # Determine class
                    if 'positive' in pred_label.lower():
                        badge_class = "sentiment-positive"
                        emoji = "😊"
                    elif 'negative' in pred_label.lower():
                        badge_class = "sentiment-negative"
                        emoji = "😞"
                    else:
                        badge_class = "sentiment-neutral"
                        emoji = "😐"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{emoji}</div>
                        <div style="font-size: 1.2rem; font-weight: 600;">Prediction</div>
                        <div style="font-size: 2rem; font-weight: 700; margin: 0.5rem 0;">
                            <span class="sentiment-badge {badge_class}" style="font-size: 1.8rem; padding: 0.5rem 2rem;">
                                {pred_label.upper()}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show probabilities
                st.markdown("#### 📊 Class Probabilities")
                
                class_names = classes if classes is not None else [f"Class {i}" for i in range(len(proba))]
                
                # Create progress bars
                for i, (name, prob) in enumerate(zip(class_names, proba)):
                    if 'positive' in name.lower():
                        bar_class = "prob-bar-positive"
                    elif 'negative' in name.lower():
                        bar_class = "prob-bar-negative"
                    else:
                        bar_class = "prob-bar-neutral"
                    
                    pct = prob * 100
                    st.markdown(f"""
                    <div style="margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-weight: 500;">{name}</span>
                            <span>{pct:.1f}%</span>
                        </div>
                        <div class="prob-bar-container">
                            <div class="prob-bar {bar_class}" style="width: {pct}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Could not make prediction. The model may not support direct prediction.")
                st.info("💡 You can still explore the model structure in other tabs.")
        else:
            st.info("💡 Enter some text above to see sentiment predictions.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Tab 4: Class Statistics ---
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📋 Class Statistics")
            
            if hasattr(model, 'class_count_') and classes is not None:
                stats_data = []
                total = model.class_count_.sum()
                
                for i, name in enumerate(classes):
                    stats_data.append({
                        "Class": name,
                        "Count": int(model.class_count_[i]),
                        "Percentage": f"{(model.class_count_[i] / total * 100):.1f}%"
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Show class prior
                if hasattr(model, 'class_log_prior_'):
                    st.markdown("#### 📐 Class Prior (Log)")
                    prior_data = []
                    for i, name in enumerate(classes):
                        prior_data.append({
                            "Class": name,
                            "Log Prior": f"{model.class_log_prior_[i]:.4f}",
                            "Prior": f"{np.exp(model.class_log_prior_[i]):.4f}"
                        })
                    prior_df = pd.DataFrame(prior_data)
                    st.dataframe(prior_df, use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ Class statistics not available.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🔢 Feature Counts")
            
            feature_counts = get_feature_counts(model)
            
            if feature_counts is not None:
                st.metric("Total Features", feature_counts.shape[1] if len(feature_counts.shape) > 1 else len(feature_counts))
                
                # Show feature count by class
                if len(feature_counts.shape) > 1 and classes is not None:
                    st.markdown("#### Feature Counts by Class")
                    for i, name in enumerate(classes):
                        if i < feature_counts.shape[0]:
                            st.metric(
                                name,
                                f"{feature_counts[i].sum():.0f}",
                                help=f"Total feature occurrences for class {name}"
                            )
            else:
                st.info("ℹ️ Feature counts not available.")
            
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Export Section ---
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if classes is not None and hasattr(model, 'class_count_'):
            export_data = {
                "Class": classes,
                "Count": model.class_count_,
                "Percentage": model.class_count_ / model.class_count_.sum() * 100
            }
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Class Statistics as CSV",
                data=csv,
                file_name="sentiment_class_stats.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6c757d; font-size: 0.85rem;'>"
        "🧠 Built with Streamlit • Multinomial Naive Bayes Sentiment Analyzer<br>"
        "Explore the structure and weights of your sentiment model"
        "</div>",
        unsafe_allow_html=True
    )

else:
    # Show instructions when no file is loaded
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📖 How to Use This App")
    
    st.markdown("""
    **This app analyzes a Multinomial Naive Bayes sentiment model pickle file. Here are the ways to load your file:**
    
    1. **📤 Upload** - Click "Browse files" and select your `sentiment_model.pkl` file
    2. **🔍 Auto-Discover** - Click the "Auto-Discover" button to search for the file in common locations
    3. **📁 Manual Placement** - Place the file in the same directory as this app and use auto-discover
    
    **After loading, you'll be able to:**
    - View class distribution and model parameters
    - Analyze feature importance for each sentiment class
    - Test the model with your own text input
    - Export class statistics as CSV
    
    **Common file locations in this project:**
    - Root directory: `/sentiment_model.pkl`
    - Sentiment Analysis: `/Sentiment_Analysis/sentiment_model.pkl`
    - Models folder: `/models/sentiment_model.pkl`
    - Data folder: `/data/sentiment_model.pkl`
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
