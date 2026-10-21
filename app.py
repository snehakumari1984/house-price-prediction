
import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="EstateIQ | House Price Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "house_price_model.pkl"
DATA_PATH = BASE_DIR / "house_data.csv"

# ---------- DARK THEME ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #202044 0%, transparent 38%),
        radial-gradient(ellipse at 90% 15%, #142e3b 0%, transparent 32%),
        #0a0b12;
    color: #f1f2fa;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f6f6ff !important;
}

p, label, .stMarkdown {
    color: #c3c6dc;
}

.hero {
    padding: 28px 30px;
    border: 1px solid #30314e;
    border-radius: 22px;
    background: linear-gradient(120deg, #17172c, #121e2c 75%, #172438);
    margin-bottom: 24px;
    box-shadow: 0 15px 45px #00000025;
}

.eyebrow {
    color: #a99bff;
    font-size: 12px;
    letter-spacing: 2.5px;
    font-weight: 700;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(30px, 4vw, 47px);
    line-height: 1.12;
    font-weight: 700;
    color: #ffffff;
    margin: 12px 0;
}

.gradient-text {
    background: linear-gradient(90deg, #a99bff, #61d9ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.muted {
    color: #9da3bd;
    font-size: 14px;
    line-height: 1.7;
}

.panel {
    background: linear-gradient(145deg, #171927, #11141f);
    border: 1px solid #2b3044;
    border-radius: 18px;
    padding: 23px;
    margin-bottom: 18px;
}

.section-label {
    color: #a99bff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.7px;
    margin-bottom: 8px;
}

div[data-testid="stMetric"] {
    background: #151827;
    border: 1px solid #2c3045;
    border-radius: 15px;
    padding: 17px 18px;
}

div[data-testid="stMetricLabel"] {
    color: #9da3bd;
}

div[data-testid="stMetricValue"] {
    color: #f5f4ff;
    font-family: 'Space Grotesk', sans-serif;
}

.stNumberInput input {
    background-color: #111522 !important;
    color: #ffffff !important;
    border-radius: 10px !important;
}

div[data-baseweb="input"] {
    border-color: #383c55 !important;
}

.stButton > button {
    width: 100%;
    min-height: 49px;
    border: 1px solid #8d7bff;
    border-radius: 11px;
    background: linear-gradient(100deg, #7765f5, #4e9cf6);
    color: white;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #c0b6ff;
    box-shadow: 0 0 24px #7968ff55;
    color: white;
}

.result-card {
    background: linear-gradient(120deg, #211c42, #142c3a);
    border: 1px solid #6d62c6;
    border-radius: 19px;
    padding: 24px;
    margin-top: 18px;
    box-shadow: 0 0 30px #6e61ff14;
}

.result-price {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(31px, 4vw, 43px);
    font-weight: 700;
    color: #a9f3ff;
    margin: 9px 0;
}

div[data-testid="stPlotlyChart"] {
    border: 1px solid #292d42;
    border-radius: 14px;
    padding: 8px;
    background: #11141f;
}

hr {
    border-color: #292d42;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

@media (max-width: 700px) {
    .hero { padding: 22px 18px; }
    .panel { padding: 17px; }
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL AND DATA ----------
if not MODEL_PATH.exists():
    st.error("Trained model not found. Run: python train_model.py")
    st.stop()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

if not DATA_PATH.exists():
    st.error("Dataset not found. Please check house_data.csv.")
    st.stop()

data = pd.read_csv(DATA_PATH)

# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <div class="eyebrow">REAL ESTATE · MACHINE LEARNING</div>
    <div class="hero-title">
        Find your home's<br>
        <span class="gradient-text">estimated value.</span>
    </div>
    <div class="muted">
        Explore property data and estimate a home's price using
        a Linear Regression model.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- METRICS ----------
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("PROPERTIES IN DATA", f"{len(data):,}")

with c2:
    st.metric(
        "AVERAGE SAMPLE PRICE",
        f"₹{data['price_lakhs'].mean():.1f} L"
    )

with c3:
    st.metric(
        "AVERAGE HOME SIZE",
        f"{data['area_sqft'].mean():,.0f} sq ft"
    )

st.markdown("---")

# ---------- MAIN LAYOUT ----------
left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown("""
    <div class="panel">
        <div class="section-label">01 / PROPERTY DETAILS</div>
        <h3 style="margin-top:0;">Configure your home</h3>
        <div class="muted">
            Enter the details below to generate an estimated price.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("property_form"):
        area = st.number_input(
            "Built-up area (sq ft)",
            min_value=100,
            max_value=10000,
            value=1200,
            step=50
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

        age = st.number_input(
            "Property age (years)",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )

        submitted = st.form_submit_button(
            "✦  Predict Property Value"
        )

    if submitted:
        input_data = pd.DataFrame(
            [[area, bedrooms, bathrooms, age]],
            columns=[
                "area_sqft",
                "bedrooms",
                "bathrooms",
                "age_years"
            ]
        )

        prediction = float(model.predict(input_data)[0])
        st.session_state["last_prediction"] = max(0, prediction)
        st.session_state["last_details"] = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age": age
        }

with right:
    st.markdown("""
    <div class="section-label">02 / YOUR ESTIMATE</div>
    <h3 style="margin-top:0;">Property valuation</h3>
    """, unsafe_allow_html=True)

    if "last_prediction" in st.session_state:
        price = st.session_state["last_prediction"]
        details = st.session_state["last_details"]

        st.markdown(f"""
        <div class="result-card">
            <div class="eyebrow">ESTIMATED PROPERTY VALUE</div>
            <div class="result-price">₹{price:.2f} Lakhs</div>
            <div class="muted">
                Based on {details['area']:,} sq ft ·
                {details['bedrooms']} bedrooms ·
                {details['bathrooms']} bathrooms ·
                {details['age']} years old
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "This is a model-generated estimate, not a verified market valuation."
        )
    else:
        st.markdown("""
        <div class="result-card">
            <div style="font-size:38px;">⌂</div>
            <h3>Your result will appear here</h3>
            <div class="muted">
                Enter your property details and click
                <b>Predict Property Value</b> to see the estimate.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
        <div class="section-label">HOW IT WORKS</div>
        <h4 style="margin:5px 0;">Linear Regression</h4>
        <div class="muted">
            The model learns relationships between property features
            and sample prices, then uses those relationships to estimate
            the price of a new input.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- DATA VISUALIZATION ----------
st.markdown("---")
st.markdown('<div class="section-label">03 / MARKET INSIGHTS</div>',
            unsafe_allow_html=True)
st.header("Explore the sample dataset")
st.markdown(
    '<div class="muted">These charts show patterns in the sample training data, '
    'not live real-estate market trends.</div>',
    unsafe_allow_html=True
)

chart1, chart2 = st.columns(2, gap="large")

with chart1:
    fig_area = px.scatter(
        data,
        x="area_sqft",
        y="price_lakhs",
        color="bedrooms",
        size="bathrooms",
        hover_data=["age_years"],
        labels={
            "area_sqft": "Area (sq ft)",
            "price_lakhs": "Price (₹ lakhs)",
            "bedrooms": "Bedrooms",
            "bathrooms": "Bathrooms",
            "age_years": "Age (years)"
        },
        template="plotly_dark",
        title="Area vs Sample Price"
    )
    fig_area.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#d9dbee",
        legend_title_text="Bedrooms",
        margin=dict(l=15, r=15, t=55, b=15)
    )
    st.plotly_chart(fig_area, use_container_width=True)

with chart2:
    bedroom_avg = (
        data.groupby("bedrooms", as_index=False)["price_lakhs"]
        .mean()
    )

    fig_bed = px.bar(
        bedroom_avg,
        x="bedrooms",
        y="price_lakhs",
        labels={
            "bedrooms": "Bedrooms",
            "price_lakhs": "Average price (₹ lakhs)"
        },
        template="plotly_dark",
        title="Average Price by Bedrooms",
        text_auto=".1f"
    )
    fig_bed.update_traces(
        marker_color="#8d7bff",
        marker_line_color="#b5aaff",
        marker_line_width=1,
        textposition="outside"
    )
    fig_bed.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#d9dbee",
        margin=dict(l=15, r=15, t=55, b=15)
    )
    st.plotly_chart(fig_bed, use_container_width=True)

with st.expander("View sample dataset"):
    st.dataframe(data, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:12px 0;">
    <div class="eyebrow">ESTATEIQ · MACHINE LEARNING PROJECT</div>
    <div class="muted" style="margin-top:8px;">
        Educational demonstration only. Predictions depend on a small,
        synthetic dataset and should not be used for buying or selling decisions.
    </div>
</div>
""", unsafe_allow_html=True)
