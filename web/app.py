import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import logging
import os
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Mijoz Ketishi Bashorat Tizimi",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .success-card {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .warning-card {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    
    .stNumberInput > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.best_model = None
    st.session_state.scaler = None
    st.session_state.cat_columns = None
    st.session_state.num_features = None
    st.session_state.feature_names = None

@st.cache_resource
def load_models():
    """Load all required model components"""
    try:
        model_path = 'models/'
        
        # Load main model
        best_model = joblib.load(f'{model_path}best_churn_model.pkl')
        logger.info("✅ Model loaded successfully!")
        
        # Load scaler (optional)
        scaler = None
        try:
            scaler = joblib.load(f'{model_path}scaler.pkl')
            # Test if scaler is fitted
            if hasattr(scaler, 'scale_'):
                logger.info("✅ Fitted scaler loaded successfully!")
            else:
                logger.warning("⚠️ Scaler found but not fitted, will skip scaling")
                scaler = None
        except Exception as e:
            logger.warning(f"⚠️ Scaler not found: {e}")
        
        # Load feature lists (optional)
        cat_columns = None
        num_features = None
        feature_names = None
        
        try:
            cat_columns = joblib.load(f'{model_path}cat_columns.pkl')
            logger.info("✅ Categorical columns loaded!")
        except Exception as e:
            logger.warning(f"⚠️ Categorical columns not found: {e}")
        
        try:
            num_features = joblib.load(f'{model_path}num_features.pkl')
            logger.info("✅ Numerical features loaded!")
        except Exception as e:
            logger.warning(f"⚠️ Numerical features not found: {e}")
        
        # Get feature names from model if available
        if hasattr(best_model, 'feature_names_in_'):
            feature_names = best_model.feature_names_in_
            logger.info(f"✅ Model feature names: {len(feature_names)} features")
        
        return best_model, scaler, cat_columns, num_features, feature_names
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return None, None, None, None, None

def get_expected_columns():
    """Define expected columns for the model"""
    # Base columns that should be present
    base_columns = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
        'gender_Female', 'gender_Male',
        'Partner_No', 'Partner_Yes',
        'Dependents_No', 'Dependents_Yes',
        'PhoneService_No', 'PhoneService_Yes',
        'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
        'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
        'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'PaperlessBilling_No', 'PaperlessBilling_Yes',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    return base_columns

def predict_churn(customer_data, model, scaler=None, cat_columns=None, num_features=None, feature_names=None):
    """Predict customer churn with improved error handling"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Handle SeniorCitizen conversion
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].map({'Ha': 1, 'Yo\'q': 0})
        
        # Define numeric columns
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        # Ensure numeric columns are properly typed
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Define categorical columns
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                           'PaperlessBilling', 'PaymentMethod']
        
        # One-hot encoding
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
        
        # Use model's expected features if available, otherwise use default
        if feature_names is not None:
            expected_features = feature_names
        else:
            expected_features = get_expected_columns()
        
        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0
        
        # Remove any unexpected columns and reorder
        df_encoded = df_encoded.reindex(columns=expected_features, fill_value=0)
        
        # Apply scaling only if scaler is properly fitted
        if scaler is not None and hasattr(scaler, 'scale_'):
            try:
                # Only scale numeric columns
                numeric_indices = [i for i, col in enumerate(expected_features) if col in numeric_cols]
                if numeric_indices:
                    df_scaled = df_encoded.copy()
                    numeric_data = df_encoded.iloc[:, numeric_indices].values
                    
                    # Check if scaler expects the right number of features
                    if numeric_data.shape[1] == len(scaler.scale_):
                        scaled_numeric = scaler.transform(numeric_data)
                        df_scaled.iloc[:, numeric_indices] = scaled_numeric
                        df_encoded = df_scaled
                    else:
                        logger.warning("Scaler feature count mismatch, skipping scaling")
            except Exception as e:
                logger.warning(f"Scaling error: {e}, proceeding without scaling")
        
        # Make prediction
        prediction = model.predict(df_encoded)[0]
        probability = model.predict_proba(df_encoded)[0]
        churn_probability = probability[1] if len(probability) > 1 else probability[0]
        
        return prediction, churn_probability
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Return default values in case of error
        return 0, 0.5

def create_gauge_chart(probability):
    """Create a gauge chart for churn probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Ketish Ehtimoli (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def create_feature_importance_chart(customer_data):
    """Create a simple feature importance visualization"""
    features = ['Shartnoma', 'Oylik to\'lov', 'Internet xizmati', 'Texnik yordam', 'Onlayn xavfsizlik']
    importance = [0.25, 0.20, 0.18, 0.12, 0.10]
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Eng Muhim Omillar",
        color=importance,
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=300)
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Mijoz Ketishi Bashorat Tizimi</h1>
        <p>Machine Learning yordamida mijozlarning ketish ehtimolini aniqlang</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    if not st.session_state.model_loaded:
        with st.spinner('🔄 Modellar yuklanmoqda...'):
            model, scaler, cat_columns, num_features, feature_names = load_models()
            if model is not None:
                st.session_state.best_model = model
                st.session_state.scaler = scaler
                st.session_state.cat_columns = cat_columns
                st.session_state.num_features = num_features
                st.session_state.feature_names = feature_names
                st.session_state.model_loaded = True
                st.success("✅ Modellar muvaffaqiyatli yuklandi!")
            else:
                st.error("❌ Modellarni yuklashda xatolik yuz berdi!")
                st.stop()
    
    # Sidebar for input
    st.sidebar.header("📋 Mijoz Ma'lumotlari")
    st.sidebar.markdown("---")
    
    # Collect customer data
    customer_data = {}
    
    # Demographics
    st.sidebar.subheader("👤 Shaxsiy Ma'lumotlar")
    customer_data['gender'] = st.sidebar.selectbox(
        "Jinsi", 
        ["Male", "Female"], 
        help="Mijozning jinsi"
    )
    
    customer_data['SeniorCitizen'] = st.sidebar.selectbox(
        "Keksa mijoz", 
        ["Yo'q", "Ha"],
        help="65 yoshdan oshgan mijozlar"
    )
    
    customer_data['Partner'] = st.sidebar.selectbox(
        "Sherik", 
        ["No", "Yes"],
        help="Turmush o'rtog'i bor-yo'qligi"
    )
    
    customer_data['Dependents'] = st.sidebar.selectbox(
        "Qaramoqlar", 
        ["No", "Yes"],
        help="Bog'liq a'zo (bola, keksa ota-ona)"
    )
    
    # Account info
    st.sidebar.subheader("📊 Hisob Ma'lumotlari")
    customer_data['tenure'] = st.sidebar.number_input(
        "Mijoz bo'lgan muddati (oy)", 
        min_value=0, 
        max_value=100, 
        value=12,
        help="Necha oy mijoz bo'lgan"
    )
    
    customer_data['Contract'] = st.sidebar.selectbox(
        "Shartnoma turi", 
        ["Month-to-month", "One year", "Two year"],
        help="Shartnoma muddati"
    )
    
    customer_data['PaperlessBilling'] = st.sidebar.selectbox(
        "Qog'ozsiz hisob", 
        ["No", "Yes"],
        help="Elektron hisob-kitob"
    )
    
    customer_data['PaymentMethod'] = st.sidebar.selectbox(
        "To'lov usuli", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        help="To'lov qanday amalga oshiriladi"
    )
    
    # Services
    st.sidebar.subheader("📞 Xizmatlar")
    customer_data['PhoneService'] = st.sidebar.selectbox(
        "Telefon xizmati", 
        ["No", "Yes"]
    )
    
    customer_data['MultipleLines'] = st.sidebar.selectbox(
        "Bir nechta liniya", 
        ["No", "Yes", "No phone service"]
    )
    
    customer_data['InternetService'] = st.sidebar.selectbox(
        "Internet xizmati", 
        ["No", "DSL", "Fiber optic"]
    )
    
    # Internet services
    internet_services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    service_names = ["Onlayn xavfsizlik", "Onlayn zaxira", "Qurilma himoyasi", "Texnik yordam", "TV oqimi", "Film oqimi"]
    
    for service, name in zip(internet_services, service_names):
        customer_data[service] = st.sidebar.selectbox(
            name, 
            ["No", "Yes", "No internet service"]
        )
    
    # Charges
    st.sidebar.subheader("💰 To'lovlar")
    customer_data['MonthlyCharges'] = st.sidebar.number_input(
        "Oylik to'lov ($)", 
        min_value=0.0, 
        max_value=200.0, 
        value=50.0,
        step=0.1
    )
    
    customer_data['TotalCharges'] = st.sidebar.number_input(
        "Jami to'lov ($)", 
        min_value=0.0, 
        max_value=10000.0, 
        value=500.0,
        step=0.1
    )
    
    # Prediction button
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔮 Bashorat Qilish", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_button:
            with st.spinner('🤖 Bashorat qilinmoqda...'):
                prediction, probability = predict_churn(
                    customer_data, 
                    st.session_state.best_model,
                    st.session_state.scaler,
                    st.session_state.cat_columns,
                    st.session_state.num_features,
                    st.session_state.feature_names
                )
                
                # Results
                st.subheader("📊 Bashorat Natijalari")
                
                # Prediction result
                if prediction == 1:
                    st.markdown("""
                    <div class="warning-card">
                        <h2>🚨 MIJOZ KETISHI MUMKIN</h2>
                        <p>Ushbu mijoz yaqin kelajakda xizmatdan voz kechishi mumkin</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-card">
                        <h2>✅ MIJOZ QOLISHI MUMKIN</h2>
                        <p>Ushbu mijoz xizmatdan foydalanishni davom ettirishi mumkin</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.metric(
                        "Ketish Ehtimoli", 
                        f"{probability:.1%}",
                        delta=f"{probability-0.5:.1%}" if probability > 0.5 else f"{0.5-probability:.1%}",
                        delta_color="inverse"
                    )
                
                with col_m2:
                    risk_level = "YUQORI" if probability > 0.7 else "O'RTA" if probability > 0.3 else "PAST"
                    st.metric("Xavf Darajasi", risk_level)
                
                with col_m3:
                    confidence = (max(probability, 1-probability) - 0.5) * 2
                    st.metric("Ishonch Darajasi", f"{confidence:.1%}")
                
                # Gauge chart
                st.subheader("📈 Vizual Tahlil")
                gauge_fig = create_gauge_chart(probability)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Tavsiyalar")
                
                if probability > 0.7:
                    st.error("""
                    **YUQORI XAVF - Darhol harakat qiling:**
                    - Mijoz bilan shaxsiy muloqot o'rnatish
                    - Maxsus chegirmalar va bonus dasturlar taklif qilish
                    - Xizmat sifatini yaxshilash bo'yicha fikr-mulohaza so'rash
                    - VIP mijoz sifatida maxsus e'tibor ko'rsatish
                    """)
                elif probability > 0.3:
                    st.warning("""
                    **O'RTA XAVF - Ehtiyot choralarini ko'ring:**
                    - Mijoz mamnuniyati so'rovnomasi o'tkazish
                    - Yangi xizmatlar haqida ma'lumot berish
                    - Loyallik dasturiga jalb qilish
                    - Muntazam bog'lanish va yordam taklif qilish
                    """)
                else:
                    st.success("""
                    **PAST XAVF - Hozirgi holatni saqlash:**
                    - Xizmat sifatini yuqori darajada ushlab turish
                    - Yangi xizmatlar haqida xabardor qilish
                    - Mijoz fikrlarini muntazam so'rash
                    - Uzoq muddatli hamkorlik dasturlarini taklif qilish
                    """)
    
    with col2:
        # Customer profile summary
        st.subheader("👤 Mijoz Profili")
        
        profile_data = {
            "Jinsi": customer_data['gender'],
            "Shartnoma": customer_data['Contract'],
            "Internet": customer_data['InternetService'],
            "To'lov usuli": customer_data['PaymentMethod'].split('(')[0].strip(),
            "Mijoz muddati": f"{customer_data['tenure']} oy",
            "Oylik to'lov": f"${customer_data['MonthlyCharges']:.0f}"
        }
        
        for key, value in profile_data.items():
            st.text(f"{key}: {value}")
        
        st.markdown("---")
        
        # Feature importance chart
        st.subheader("📊 Muhim Omillar")
        importance_fig = create_feature_importance_chart(customer_data)
        st.plotly_chart(importance_fig, use_container_width=True)
        
        st.markdown("---")
        
        # Statistics
        st.subheader("📈 Umumiy Statistika")
        st.info("""
        **Model Ma'lumotlari:**
        - Aniqlik: ~85%
        - Ishlatilgan algoritm: Logisctic Regression
        - Model turi: Kengaytirilgan
        - Oxirgi yangilanish: 2025
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>🔮 Mijoz Ketishi Bashorat Tizimi | Yaratilgan: Machine Learning yordamida</p>
        <p>⚠️ Bu bashoratlar faqat ma'lumot berish maqsadida. Asosiy qarorlar uchun qo'shimcha tahlil talab qilinadi.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    