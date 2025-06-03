import streamlit as st
import pandas as pd
import joblib
import time
from datetime import datetime
import plotly.express as px
from PIL import Image

# Configuration
st.set_page_config(
    page_title="Flight Demand Forecast",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logo
logo = Image.open("logo.png")
st.image(logo, width=160)
# Theme selection
theme_choice = st.sidebar.selectbox("🎨 Select Theme", ["Light", "Dark"])
# Theme-specific base styles (background + text color)
light_base_css = """
    <style>
        .main, .block-container {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
    </style>
"""

dark_base_css = """
    <style>
        .main, .block-container {
            background-color: #121212 !important;
            color: #f0f0f0 !important;
        }
    </style>
"""

# Apply the selected base theme
if theme_choice == "Light":
    st.markdown(light_base_css, unsafe_allow_html=True)
else:
    st.markdown(dark_base_css, unsafe_allow_html=True)

# Custom CSS
st.markdown("""
            
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
            font-weight: bold;
            border: none;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        .prediction-card {
            background: linear-gradient(135deg, #6e8efb, #a777e3);
            color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .history-card {
            background-color: var(--background-color, #ffffff);
            color: var(--text-color, #000000);
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Load models and encodings
@st.cache_resource
def load_models():
    model = joblib.load("model_vols_prediction.pkl")
    freq_encoding = joblib.load("freq_encoding_pays.pkl")
    model_features = joblib.load("model_features.pkl")
    return model, freq_encoding, model_features

model, freq_encoding, model_features = load_models()

# Initialize history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Sidebar with reset
with st.sidebar:
    st.title("🗕️ Prediction History")
    if st.button("🗑️ Reset History"):
        st.session_state.prediction_history = []
        st.experimental_rerun()

    if st.session_state.prediction_history:
        for i, entry in enumerate(reversed(st.session_state.prediction_history[-5:])):
            st.markdown(f"""
                <div class="history-card">
                    <b>Prediction #{len(st.session_state.prediction_history)-i}</b><br>
                    <small>{entry['timestamp']}</small><br>
                    <b>{entry['prediction']:.0f} flights</b> to {entry['country']}<br>
                    Season: {entry['season']}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No predictions yet. Make your first prediction!")

# Main title
st.title("✈️ Flight Demand Forecasting Dashboard")
st.markdown("Predict seasonal flight demand based on market conditions and historical data.")

# Input form
with st.form("prediction_form"):
    st.subheader("📋 Input Parameters")
    col1, col2 = st.columns(2)
    with col1:
        saison = st.selectbox("🗓️ Season", ["Winter", "Spring", "Summer", "Autumn"])
        pays = st.selectbox("🌍 Destination Country", sorted(freq_encoding.index.tolist()))
        vols_prec = st.slider("🔁 Previous season flights", 0, 1000, 100)
        tarif = st.number_input("💸 Average price per person (€)", 0, 10000, 2500)
    with col2:
        prix_vente = st.number_input("💰 Average selling price (€)", 0, 10000, 3000)
        stock = st.number_input("📦 Average available stock", 0, 10000, 200)
        prix_achat = st.number_input("🛒 Average purchase price (€)", 0, 10000, 2500)
        commission = st.number_input("📊 Average commission (%)", 0.0, 100.0, 10.0, step=0.5)
    col3, col4 = st.columns(2)
    with col3:
        montant = st.number_input("💶 Average amount paid (€)", 0, 10000, 2700)
    with col4:
        nb_passagers = st.slider("👨‍👩‍👦 Number of passengers", 0, 1000, 150)

    aller_retour_rate = st.slider("🔁 Round-trip rate", 0.0, 1.0, 0.5, step=0.05)
    submitted = st.form_submit_button("🚀 Predict Flight Demand")

# Prediction logic
if submitted:
    with st.spinner('🔮 Predicting flight demand...'):
        time.sleep(1)
        saison_enc = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}[saison]
        pays_freq = freq_encoding.get(pays, 0.0)

        input_dict = {
            "saison_enc": saison_enc,
            "pays_freq": pays_freq,
            "vols_saison_dernière": vols_prec,
            "tarif_moyen_grille": tarif,
            "prix_vente_moyen": prix_vente,
            "prix_achat_moyen": prix_achat,
            "commission_moyenne": commission,
            "montant_moyen": montant,
            "nb_passagers": nb_passagers,
            "aller_retour_rate": aller_retour_rate,
            "stock_moyen_grille": stock
        }

        X_input = pd.DataFrame([[input_dict[f] for f in model_features]], columns=model_features)
        prediction = model.predict(X_input)[0]

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "prediction": prediction,
            "country": pays,
            "season": saison,
            "input_data": input_dict
        }
        st.session_state.prediction_history.append(entry)

        st.markdown("---")
        st.subheader("📊 Prediction Results")
        st.markdown(f"""
            <div class='prediction-card'>
                <h3 style='color:white'>Forecasted Flight Demand</h3>
                <h1 style='text-align:center;'>{prediction:.0f} flights</h1>
                <p style='text-align:center;'>Season: {saison} | Destination: {pays} | Previous: {vols_prec}</p>
            </div>
        """, unsafe_allow_html=True)

        # Summary
        st.info(f"📌 {entry['timestamp']} → {saison} → {pays}: **{prediction:.0f} flights**")

        # Badge
        if prediction < 100:
            st.warning("🔍 Low demand expected.")
        elif prediction < 500:
            st.success("✅ Moderate demand forecasted.")
        else:
            st.error("🚨 High demand forecasted.")

        # Export CSV
        history_df = pd.DataFrame(st.session_state.prediction_history)
        with st.expander("📤 Export history as CSV"):
            st.download_button("⬇️ Download", data=history_df.to_csv(index=False),
                               file_name="flight_predictions.csv", mime="text/csv")

        # Plot
        if len(history_df) > 1:
            st.subheader("📈 Prediction Trends")
            history_df['timestamp_dt'] = pd.to_datetime(history_df['timestamp'])
            fig = px.line(history_df, x='timestamp_dt', y='prediction', color='country', markers=True,
                          labels={'prediction': 'Predicted Flights', 'timestamp_dt': 'Time'},
                          title='Flight Demand Prediction History')
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        # Inputs
        with st.expander("📋 View input data used"):
            st.dataframe(X_input, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: grey; padding: 10px;">
        Flight Demand Forecasting System • Powered by Machine Learning
    </div>
""", unsafe_allow_html=True)
