import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os

# Set page config for a wide, premium layout
st.set_page_config(
    page_title="FraudShield AI - Real-time Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-end Glassmorphic & Modern Styling
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');
    
    /* Font overrides */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B, #FF8F8F, #8F9DFF, #4B6BFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #A0AEC0;
        margin-bottom: 2rem;
    }
    
    /* Premium Glassmorphic Cards & Containers */
    .glass-card, div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(17, 24, 39, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Fix Safari image collapsing bug in columns */
    div[data-testid="stImage"] img {
        width: 100% !important;
        height: auto !important;
        display: block !important;
    }
    
    /* Status Badge styling */
    .status-badge {
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-loaded {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-missing {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    /* Quick Test buttons style */
    .btn-container {
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
    }
    
    /* Result styling */
    .secure-result {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.2));
        border: 1px solid #10B981;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    .secure-title {
        color: #10B981;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .fraud-result {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.2));
        border: 1px solid #EF4444;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);
    }
    .fraud-title {
        color: #EF4444;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    /* Metric card styles */
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        font-family: 'Space Grotesk', sans-serif;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Define directories
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "model_outputs"
DATA_PATH = BASE_DIR / "creditcard_2023.csv"

# Model human names to keys map
MODEL_MAP = {
    "XGBoost Classifier": "xgboost",
    "LightGBM Classifier": "lightgbm",
    "Multi-Layer Perceptron (Neural Network)": "neural_network",
    "Logistic Regression": "logistic_regression",
    "Decision Tree Classifier": "decision_tree",
    "Support Vector Machine (Linear SVC)": "svm"
}

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("<div style='text-align: center; padding: 10px 0;'><h2 style='font-family:\"Space Grotesk\"; margin: 0; color: #FFFFFF;'>🛡️ FraudShield AI</h2><p style='color:#718096; font-size:0.9rem; margin-top:5px;'>Version 1.0.0</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.subheader("Model Configuration")
selected_model_name = st.sidebar.selectbox("Active Classifier Model", list(MODEL_MAP.keys()))
model_key = MODEL_MAP[selected_model_name]

# Check if model joblib is available
model_file_name = f"{model_key}_model.joblib"
model_path = OUTPUT_DIR / model_file_name
model_loaded = False

if model_path.exists():
    try:
        model = joblib.load(model_path)
        model_loaded = True
        st.sidebar.markdown(f'<div class="status-badge status-loaded">🟢 {selected_model_name} Loaded</div>', unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
else:
    st.sidebar.markdown(f'<div class="status-badge status-missing">🟡 {selected_model_name} training/not found</div>', unsafe_allow_html=True)
    st.sidebar.warning(f"Please wait. The background training task is saving `{model_file_name}`. Showing pre-computed metrics summary in the meantime.")

# Quick Model Info
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
model_descriptions = {
    "xgboost": "XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.",
    "lightgbm": "LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient with fast training speeds.",
    "neural_network": "A Multi-layer Perceptron (MLP) Artificial Neural Network with 32 hidden layers. Trained on a balanced subset of 50k transactions with standard scaling.",
    "logistic_regression": "A baseline linear model that computes the log-odds of a transaction being fraudulent. Highly interpretable, lightweight, and extremely fast.",
    "decision_tree": "A non-linear model that splits the feature space into hierarchical decisions. Highly interpretable tree of depth 10.",
    "svm": "A Linear Support Vector Classifier that fits a hyperplane to separate secure and fraudulent transactions. Regularized with standard scaling."
}
st.sidebar.info(model_descriptions[model_key])

# ----------------- DATA CACHING -----------------
@st.cache_data
def load_csv_samples(file_path):
    """Loads a balanced subset of transactions from the massive 324MB CSV file for quick testing."""
    if not file_path.exists():
        return None, None
    try:
        # Since reading a 324MB CSV takes ~3 seconds, cache it
        df = pd.read_csv(file_path)
        
        # Pull 50 random samples of each class to populate UI buttons quickly
        safe = df[df["Class"] == 0].sample(50, random_state=42)
        fraud = df[df["Class"] == 1].sample(50, random_state=42)
        
        # Drop columns not used as features
        features_to_drop = ["Class", "id"]
        safe_features = safe.drop(columns=[col for col in features_to_drop if col in safe.columns])
        fraud_features = fraud.drop(columns=[col for col in features_to_drop if col in fraud.columns])
        
        return safe_features, fraud_features
    except Exception as e:
        st.warning(f"Unable to preload real transactions from creditcard_2023.csv: {e}")
        return None, None

# Load testing samples
safe_samples, fraud_samples = load_csv_samples(DATA_PATH)

# Initialize Session State for individual feature values if not already present
if "manual_features" not in st.session_state:
    st.session_state["manual_features"] = {f"V{i}": 0.0 for i in range(1, 29)}
    st.session_state["manual_features"]["Amount"] = 100.0

# Helper function to reset/load transaction into session state
def populate_fields(source_df):
    if source_df is not None:
        random_row = source_df.sample(1).iloc[0]
        for key in st.session_state["manual_features"].keys():
            if key in random_row:
                st.session_state["manual_features"][key] = float(random_row[key])

# ----------------- MAIN LAYOUT -----------------
st.markdown("<h1 class='main-title'>🛡️ FraudShield AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict, analyze, and inspect credit card transactions using state-of-the-art Machine Learning models.</p>", unsafe_allow_html=True)

# Tabs definition
tab1, tab2, tab3 = st.tabs([
    "📊 Model Performance Explorer", 
    "🎯 Single-Transaction Predictor", 
    "📁 Batch CSV Predictor"
])

# ================= TAB 1: MODEL PERFORMANCE EXPLORER =================
with tab1:
    st.markdown("### 📈 Evaluation & Performance Dashboard")
    
    # Load metrics summary
    metrics_path = OUTPUT_DIR / "metrics_summary.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        
        # Display global comparison summary
        with st.container(border=True):
            st.subheader("📊 Comparative Summary of All Classifier Models")
            st.dataframe(
                metrics_df.style.format({
                    "accuracy": "{:.5f}",
                    "precision": "{:.5f}",
                    "recall": "{:.5f}",
                    "f1_score": "{:.5f}",
                    "roc_auc": "{:.5f}",
                    "pr_auc": "{:.5f}"
                }),
                use_container_width=True
            )
        
        # Display selected model metrics
        model_row = metrics_df[metrics_df["model"] == model_key]
        if not model_row.empty:
            model_metrics = model_row.iloc[0]
            st.markdown(f"#### 🛡️ Selected Model: {selected_model_name}")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class='glass-card' style='text-align: center;'>
                    <div class='metric-lbl'>Accuracy</div>
                    <div class='metric-val'>{model_metrics['accuracy']:.4%}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='glass-card' style='text-align: center;'>
                    <div class='metric-lbl'>F1-Score</div>
                    <div class='metric-val'>{model_metrics['f1_score']:.4%}</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class='glass-card' style='text-align: center;'>
                    <div class='metric-lbl'>ROC AUC</div>
                    <div class='metric-val'>{model_metrics['roc_auc']:.5f}</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class='glass-card' style='text-align: center;'>
                    <div class='metric-lbl'>PR AUC</div>
                    <div class='metric-val'>{model_metrics['pr_auc']:.5f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            c5, c6 = st.columns(2)
            with c5:
                st.markdown(f"""
                <div class='glass-card' style='text-align: center;'>
                    <div class='metric-lbl'>Precision</div>
                    <div class='metric-val'>{model_metrics['precision']:.4%}</div>
                </div>
                """, unsafe_allow_html=True)
            with c6:
                st.markdown(f"""
                <div class='glass-card' style='text-align: center;'>
                    <div class='metric-lbl'>Recall (Sensitivity)</div>
                    <div class='metric-val'>{model_metrics['recall']:.4%}</div>
                </div>
                """, unsafe_allow_html=True)
                
        # Side-by-side pre-computed plots
        st.markdown("### 🧪 Performance Curves & Confusion Matrices")
        plot1, plot2, plot3 = st.columns(3)
        
        roc_plot = OUTPUT_DIR / f"{model_key}_roc_curve.png"
        pr_plot = OUTPUT_DIR / f"{model_key}_pr_curve.png"
        cm_plot = OUTPUT_DIR / f"{model_key}_confusion_matrix.png"
        
        with plot1:
            with st.container(border=True):
                st.markdown("**ROC (Receiver Operating Characteristic) Curve**")
                if roc_plot.exists():
                    st.image(str(roc_plot), use_container_width=True)
                else:
                    st.warning("ROC Curve plot not found.")
            
        with plot2:
            with st.container(border=True):
                st.markdown("**Precision-Recall Curve**")
                if pr_plot.exists():
                    st.image(str(pr_plot), use_container_width=True)
                else:
                    st.warning("Precision-Recall Curve plot not found.")
            
        with plot3:
            with st.container(border=True):
                st.markdown("**Confusion Matrix**")
                if cm_plot.exists():
                    st.image(str(cm_plot), use_container_width=True)
                else:
                    st.warning("Confusion Matrix plot not found.")
    else:
        st.info("Metrics summary file `metrics_summary.csv` not found. Please train models or wait for training to complete.")


# ================= TAB 2: SINGLE-TRANSACTION PREDICTOR =================
with tab2:
    st.markdown("### 🔮 Real-Time Single Transaction Prediction")
    st.markdown("Input the transaction amount and the V1-V28 PCA (Principal Components) values below. Use the quick-loaders to pull real data from the dataset instantly.")
    
    # Quick testing buttons
    with st.container(border=True):
        st.markdown("##### 🚀 Quick Scenario Loaders")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
        
        with col_btn1:
            if st.button("🟢 Load Random Safe Case", use_container_width=True):
                if safe_samples is not None:
                    populate_fields(safe_samples)
                    st.toast("Real Legitimate Transaction features loaded!", icon="🟢")
                else:
                    st.error("Real dataset samples not loaded.")
                    
        with col_btn2:
            if st.button("🔴 Load Random Fraud Case", use_container_width=True):
                if fraud_samples is not None:
                    populate_fields(fraud_samples)
                    st.toast("Real Fraudulent Transaction features loaded!", icon="🔴")
                else:
                    st.error("Real dataset samples not loaded.")
                    
        with col_btn3:
            if st.button("🔄 Reset to Neutral (Zeros)", use_container_width=True):
                for k in st.session_state["manual_features"].keys():
                    st.session_state["manual_features"][k] = 0.0
                st.session_state["manual_features"]["Amount"] = 100.0
                st.toast("Inputs reset to default values!")

    # Prediction Inputs organized in expanders
    st.markdown("### 🎛️ Transaction Parameters")
    
    col_inputs1, col_inputs2 = st.columns(2)
    
    with col_inputs1:
        with st.expander("💸 Transaction Details & PCA V1 - V14", expanded=True):
            st.session_state["manual_features"]["Amount"] = st.number_input(
                "Transaction Amount ($)", 
                value=st.session_state["manual_features"]["Amount"], 
                step=10.0, 
                min_value=0.0
            )
            for i in range(1, 15):
                st.session_state["manual_features"][f"V{i}"] = st.number_input(
                    f"Component V{i}", 
                    value=st.session_state["manual_features"][f"V{i}"], 
                    format="%.6f", 
                    step=0.1
                )
                
    with col_inputs2:
        with st.expander("🛡️ PCA Components V15 - V28", expanded=True):
            for i in range(15, 29):
                st.session_state["manual_features"][f"V{i}"] = st.number_input(
                    f"Component V{i}", 
                    value=st.session_state["manual_features"][f"V{i}"], 
                    format="%.6f", 
                    step=0.1
                )

    st.markdown("---")
    
    # Run prediction
    if st.button("🔍 Analyze Transaction Integrity", type="primary", use_container_width=True):
        if not model_loaded:
            st.error(f"Cannot run prediction: Model {selected_model_name} is not loaded.")
        else:
            # Prepare feature vector (V1-V28, Amount)
            feature_keys = [f"V{i}" for i in range(1, 29)] + ["Amount"]
            input_values = [st.session_state["manual_features"][k] for k in feature_keys]
            input_df = pd.DataFrame([input_values], columns=feature_keys)
            
            # Predict
            pred_class = model.predict(input_df)[0]
            
            # Probability calculation (if supported)
            prob_fraud = None
            prob_safe = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                prob_safe = probs[0]
                prob_fraud = probs[1]
            elif hasattr(model, "decision_function"):
                decision_score = model.decision_function(input_df)[0]
                # Sigmoid scaling for decision function score
                prob_fraud = 1 / (1 + np.exp(-decision_score))
                prob_safe = 1 - prob_fraud

            # Output results
            if pred_class == 0:
                # Legitimate
                safe_pct = f"{prob_safe:.4%}" if prob_safe is not None else "High"
                st.markdown(f"""
                <div class="secure-result">
                    <div class="secure-title">🛡️ TRANSACTION SECURE</div>
                    <p style="font-size:1.1rem; color:#FFFFFF; margin-bottom:5px;">This transaction is categorized as <b>LEGITIMATE</b>.</p>
                    <p style="color:#A0AEC0; margin-bottom:0;">Safety Confidence: <b>{safe_pct}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Fraudulent
                fraud_pct = f"{prob_fraud:.4%}" if prob_fraud is not None else "High"
                st.markdown(f"""
                <div class="fraud-result">
                    <div class="fraud-title">⚠️ FRAUDULENT ACTIVITY DETECTED</div>
                    <p style="font-size:1.1rem; color:#FFFFFF; margin-bottom:5px;">This transaction has high-risk features resembling <b>FRAUD</b>.</p>
                    <p style="color:#FCA5A5; font-size: 1.1rem; margin-bottom:0;">Fraud Probability: <b>{fraud_pct}</b></p>
                </div>
                """, unsafe_allow_html=True)


# ================= TAB 3: BATCH CSV PREDICTOR =================
with tab3:
    st.markdown("### 📁 Batch Transaction Processing")
    st.markdown("Upload a CSV file containing multiple credit card transaction records. The file must contain columns `V1` to `V28` and `Amount` (columns like `id` and `Class` are optional and will be handled automatically).")
    
    # Offer downlad of sample template
    if safe_samples is not None and fraud_samples is not None:
        template_df = pd.concat([safe_samples.head(5), fraud_samples.head(5)]).reset_index(drop=True)
        # Convert to CSV for download
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=template_csv,
            file_name="credit_card_batch_sample.csv",
            mime="text/csv",
            help="Download a balanced sample template with 10 actual records to test the uploader."
        )
        
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Transaction CSV File", type=["csv"])
    
    if uploaded_file is not None:
        if not model_loaded:
            st.error(f"Cannot run predictions: Model {selected_model_name} is not loaded.")
        else:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                
                # Check for necessary features
                req_features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
                missing = [col for col in req_features if col not in batch_df.columns]
                
                if missing:
                    st.error(f"Upload fails: The CSV file is missing the following required columns: {', '.join(missing)}")
                else:
                    # Run predictions
                    prediction_features = batch_df[req_features]
                    predictions = model.predict(prediction_features)
                    
                    # Store predictions
                    result_df = batch_df.copy()
                    result_df["Fraud_Prediction"] = predictions
                    result_df["Prediction_Label"] = result_df["Fraud_Prediction"].map({0: "Secure", 1: "Fraud"})
                    
                    # Compute probabilities if available
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(prediction_features)
                        result_df["Fraud_Probability"] = probs[:, 1]
                    elif hasattr(model, "decision_function"):
                        dec_scores = model.decision_function(prediction_features)
                        result_df["Fraud_Probability"] = 1 / (1 + np.exp(-dec_scores))
                    else:
                        result_df["Fraud_Probability"] = np.nan
                        
                    # Summary metrics of the batch
                    total_rows = len(result_df)
                    fraud_count = int(np.sum(predictions))
                    safe_count = total_rows - fraud_count
                    fraud_rate = fraud_count / total_rows
                    
                    st.markdown("### 📊 Batch Prediction Results Summary")
                    bc1, bc2, bc3 = st.columns(3)
                    
                    with bc1:
                        st.markdown(f"""
                        <div class='glass-card' style='text-align: center;'>
                            <div class='metric-lbl'>Total Processed</div>
                            <div class='metric-val'>{total_rows}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with bc2:
                        st.markdown(f"""
                        <div class='glass-card' style='text-align: center; border-color: rgba(16, 185, 129, 0.4);'>
                            <div class='metric-lbl' style='color: #10B981;'>Secure Transactions</div>
                            <div class='metric-val' style='color: #10B981;'>{safe_count}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with bc3:
                        st.markdown(f"""
                        <div class='glass-card' style='text-align: center; border-color: rgba(239, 68, 68, 0.4);'>
                            <div class='metric-lbl' style='color: #EF4444;'>Fraud Warnings</div>
                            <div class='metric-val' style='color: #EF4444;'>{fraud_count} ({fraud_rate:.2%})</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Preview predicted dataset
                    st.markdown("#### 🔍 Annotated Transaction Data Preview")
                    
                    # Highlight fraud rows in the table
                    def highlight_fraud(row):
                        return ['background-color: rgba(239, 68, 68, 0.15); color: #EF4444;' if row.Fraud_Prediction == 1 else '' for _ in row]
                    
                    # Order columns nicely
                    cols_order = ["Prediction_Label", "Fraud_Probability"] + req_features
                    if "Class" in result_df.columns:
                        cols_order = ["Prediction_Label", "Class", "Fraud_Probability"] + req_features
                    
                    st.dataframe(
                        result_df[cols_order].style.apply(highlight_fraud, axis=1)
                        .format({
                            "Fraud_Probability": "{:.4%}"
                        }),
                        use_container_width=True
                    )
                    
                    # Download predicted results
                    result_csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Complete Predictions CSV",
                        data=result_csv,
                        file_name="transaction_predictions_annotated.csv",
                        mime="text/csv",
                        type="primary"
                    )
            except Exception as e:
                st.error(f"Error processing CSV file: {e}")
