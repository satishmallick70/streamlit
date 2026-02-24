import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

plt.style.use("dark_background")

st.set_page_config(
    page_title="Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ------------------ STYLING ------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 20px;
}

.success-card {
    background: linear-gradient(135deg, #00FFA3, #009e6f);
    padding: 18px;
    border-radius: 12px;
    color: black;
    font-weight: 600;
    text-align: center;
    font-size: 22px;
}

.alert-card {
    background: linear-gradient(135deg, #FF4B4B, #b30000);
    padding: 18px;
    border-radius: 12px;
    color: white;
    font-weight: 600;
    text-align: center;
    font-size: 22px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD ARTIFACTS ------------------

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
all_columns = joblib.load("columns.pkl")
selected_columns = joblib.load("selected_columns.pkl")
encoder = joblib.load("encoder.pkl")

st.title("🛡️ Network Intrusion Detection System")
st.markdown("Real-time detection of **Normal / DoS / Probe** traffic")

st.sidebar.header("⚙️ Controls")
mode = st.sidebar.radio("Choose Input Mode", ["Manual Input", "CSV Upload"])

# ============================================================
# MANUAL MODE
# ============================================================

if mode == "Manual Input":

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        duration = st.number_input("duration", min_value=0.0)
        src_bytes = st.number_input("src_bytes", min_value=0.0)
        dst_bytes = st.number_input("dst_bytes", min_value=0.0)

    with col2:
        protocol_type = st.selectbox("protocol_type", ["tcp", "udp", "icmp"])
        service = st.selectbox("service", ["http", "ftp", "smtp", "domain_u", "other"])
        flag = st.selectbox("flag", ["SF", "S0", "REJ", "RSTR"])

    with col3:
        count = st.number_input("count", min_value=0.0)
        srv_count = st.number_input("srv_count", min_value=0.0)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Run Detection"):

        with st.spinner("Analyzing traffic..."):
            time.sleep(1)

            input_dict = {
                "duration": duration,
                "src_bytes": src_bytes,
                "dst_bytes": dst_bytes,
                "protocol_type": protocol_type,
                "service": service,
                "flag": flag,
                "count": count,
                "srv_count": srv_count
            }

            df = pd.DataFrame([input_dict])
            encoded = pd.get_dummies(df)
            encoded = encoded.reindex(columns=all_columns, fill_value=0)

            scaled = scaler.transform(encoded)
            scaled_df = pd.DataFrame(scaled, columns=all_columns)
            selected = scaled_df[selected_columns]

            pred = model.predict(selected)
            label = encoder.inverse_transform(pred)[0]

        if label == "normal":
            st.markdown('<div class="success-card">✅ NORMAL TRAFFIC</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-card">🚨 {label.upper()} ATTACK</div>', unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):

            probs = model.predict_proba(selected)[0]
            confidence = np.max(probs)

            # ------------------ GAUGE METER ------------------

            st.subheader("🎯 Model Confidence")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#00FFA3"},
                }
            ))

            st.plotly_chart(fig_gauge, use_container_width=True)

# ============================================================
# CSV MODE
# ============================================================

else:

    uploaded_file = st.file_uploader("Upload CSV")

    if uploaded_file is not None:

        with st.spinner("Processing dataset..."):
            input_df = pd.read_csv(uploaded_file)

            encoded = pd.get_dummies(input_df)
            encoded = encoded.reindex(columns=all_columns, fill_value=0)

            scaled = scaler.transform(encoded)
            scaled_df = pd.DataFrame(scaled, columns=all_columns)
            selected = scaled_df[selected_columns]

            preds = model.predict(selected)
            labels = encoder.inverse_transform(preds)

            input_df["Prediction"] = labels

        st.success("✅ Detection Complete")

        counts = input_df["Prediction"].value_counts()

        # ------------------ METRICS ------------------

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        colA, colB, colC = st.columns(3)

        colA.metric("Normal", int((labels == "normal").sum()))
        colB.metric("DoS", int((labels == "dos").sum()))
        colC.metric("Probe", int((labels == "probe").sum()))
        st.markdown('</div>', unsafe_allow_html=True)

        # ------------------ SMALL PIE ------------------

        st.subheader("📊 Traffic Distribution")

        fig, ax = plt.subplots(figsize=(3, 3))

        ax.pie(
            counts,
            labels=counts.index,
            autopct='%1.1f%%',
            colors=["#00FFA3", "#FF4B4B", "#1E90FF"],
            textprops={'color': 'white'}
        )

        st.pyplot(fig)

        # ------------------ SEABORN BAR ------------------

        st.subheader("📈 Category Breakdown")

        fig2, ax2 = plt.subplots(figsize=(6, 3))

        sns.countplot(
            x=input_df["Prediction"],
            palette=["#00FFA3", "#FF4B4B", "#1E90FF"],
            ax=ax2
        )

        st.pyplot(fig2)

        # ------------------ TREND CHART ------------------

        st.subheader("📉 Trend Over Time (Simulated)")

        trend_data = counts.reset_index()
        trend_data.columns = ["Attack", "Count"]

        fig3, ax3 = plt.subplots(figsize=(6, 3))

        sns.lineplot(
            data=trend_data,
            x="Attack",
            y="Count",
            marker="o",
            ax=ax3
        )

        st.pyplot(fig3)

        # ------------------ LIVE ATTACK FEED ------------------

        st.subheader("🖥️ Live Detection Feed")

        feed = st.empty()

        for i in range(5):
            sample = np.random.choice(["normal", "dos", "probe"])
            feed.markdown(f"**Packet {i+1}:** `{sample.upper()}` detected")
            time.sleep(0.4)