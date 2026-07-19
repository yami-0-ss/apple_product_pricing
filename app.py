import streamlit as st
import pickle
import numpy as np

# -------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="Gadget Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Attractive CSS Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #ff4b4b;
        text-align: center;
        margin: 10px 0;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# -------------------------------------------------------------
# SIDEBAR - INPUT FEATURES
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/clouds/100/000000/smartphone.png", width=100)
st.sidebar.title("Specifications")
st.sidebar.markdown("Adjust device parameters below:")

# --- Adjust these sliders/inputs to match your true model features ---
spec_1 = st.sidebar.slider("Feature Name 1 (e.g., RAM GB)", min_value=2, max_value=16, value=8)
spec_2 = st.sidebar.slider("Feature Name 2 (e.g., Storage GB)", min_value=32, max_value=512, value=128)
spec_3 = st.sidebar.selectbox("Feature Name 3 (e.g., Brand Rating)", [1, 2, 3, 4, 5], index=3)
# ---------------------------------------------------------------------

# -------------------------------------------------------------
# MAIN DASHBOARD LAYOUT
# -------------------------------------------------------------
st.title("📱 Smart Device Valuation Dashboard")
st.markdown("Predict the **Launch Price** of tech hardware using machine learning insights.")
st.write("---")

# Predict Logic
# Prepare input payload mapping to how your model was trained
input_data = np.array([[spec_1, spec_2, spec_3]]) 

if st.sidebar.button("✨ Calculate Estimated Value", use_container_width=True):
    with st.spinner('Analyzing valuation metrics...'):
        try:
            # Generate predictions
            prediction = model.predict(input_data)[0]
            
            # Assuming output contains standard regression array mapping 
            # If your model predicts both USD and INR dynamically:
            pred_usd = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
            pred_inr = prediction[1] if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 1 else pred_usd * 83.5 

            # Display Output Cards side-by-side
            st.subheader("💡 Predicted Market Launch Estimates")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-lbl">Estimated Price (USD)</div>
                        <div class="metric-val">${pred_usd:,.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                    <div class="metric-box" style="border-left-color: #10b981;">
                        <div class="metric-lbl">Estimated Price (INR)</div>
                        <div class="metric-val">₹{pred_inr:,.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            st.balloons()

        except Exception as e:
            st.error(f"Prediction failed. Ensure feature sequence matches data shape: {e}")
else:
    st.info("👈 Set the hardware specifications in the sidebar panel and click **Calculate Estimated Value**.")
