import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Transport Delay Dashboard", page_icon="🚌", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;600;700&display=swap');
.stApp {background-color: #0a0a0a;color: #fafafa;font-family: 'Manrope', sans-serif;}
@keyframes fadeIn {from {opacity: 0; transform: translateY(10px);}to {opacity: 1; transform: translateY(0);}}
@keyframes pulse {0%, 100% {opacity: 1;}50% {opacity: 0.7;}}
.header {border: 1px solid rgba(255,255,255,0.1);background: rgba(255,255,255,0.02);border-radius: 20px;padding: 2.2rem 1.2rem;text-align: center;margin-bottom: 2rem;animation: fadeIn 0.8s ease-in-out;}
.header h1 {font-size: 2.3rem;letter-spacing: -1px;font-weight: 700;color: #fafafa;}
.header p {font-size: 1rem;color: #a1a1a1;}
.metric-card {display: flex;flex-direction: column;align-items: center;justify-content: center;border: 1px solid rgba(255,255,255,0.1);background: rgba(255,255,255,0.02);border-radius: 16px;padding: 1.2rem;margin: 0.5rem;text-align: center;animation: fadeIn 0.8s ease-in-out;}
.metric-value {font-size: 2rem;font-weight: 700;color: white;}
.metric-label {font-size: 0.9rem;color: #a1a1a1;margin-top: 0.3rem;}
.insight-box {border: 1px solid rgba(255,255,255,0.1);background: rgba(255,255,255,0.04);padding: 1rem;border-radius: 12px;color: #e5e5e5;margin-top: 0.5rem;}
.stTabs [data-baseweb="tab-list"] {justify-content: center;gap: 25px;}
.stTabs [data-baseweb="tab"] {background-color: transparent;color: #fafafa;border: 1px solid rgba(255,255,255,0.1);border-radius: 12px;padding: 0.6rem 1.4rem;transition: all 0.3s ease-in-out;}
.stTabs [data-baseweb="tab"]:hover {border-color: #ffffff;}
.stTabs [aria-selected="true"] {background: white;color: #0a0a0a !important;font-weight: 600;animation: pulse 2s infinite;}
[data-testid="stDataFrame"] {background: rgba(255,255,255,0.03);border: 1px solid rgba(255,255,255,0.1);border-radius: 12px;box-shadow: 0 0 10px rgba(255,255,255,0.05);}
div[data-testid="stPlotlyChart"], div[data-testid="stPyplotChart"] {animation: fadeIn 0.6s ease-in-out;border-radius: 12px;}
div[data-testid="stNotification"] {border-radius: 12px !important;font-weight: 500 !important;padding: 1rem !important;}
div[data-testid="stNotification"][role="alert"] {background-color: rgba(255,255,255,0.05) !important;color: #ffffff !important;border-left: 4px solid #fff !important;}
.footer {text-align: center;color: #888;font-size: 0.9rem;margin-top: 2rem;padding-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
  <h1>🚌 Transport Delay Dashboard</h1>
  <p>Real-time performance overview for public transport</p>
</div>
""", unsafe_allow_html=True)

try:
    model = joblib.load("xgb_model.joblib")
    meta = joblib.load("meta.joblib")
except Exception as e:
    st.error(f"Could not load model or metadata: {e}")
    st.stop()

df_live = pd.DataFrame()

st.sidebar.markdown("### ⚙️ Configuration Panel")

upload_choice = st.sidebar.radio(
    "📦 Data Source",
    ["Use Default Live Feed", "Upload Custom Dataset"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗓️ Time Window")
date_option = st.sidebar.selectbox(
    "Select Date Range",
    ["All Time", "Last 24 Hours", "Last 7 Days", "Last 30 Days"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚍 Route Focus")
try:
    all_routes = sorted(df_live["route_id"].unique().tolist())
except Exception:
    all_routes = [f"R{i}" for i in range(1, 6)]
selected_route_sidebar = st.sidebar.selectbox(
    "Highlight Route",
    ["All Routes"] + all_routes
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Info")
try:
    rows = len(df_live)
    last_time = pd.to_datetime(df_live["timestamp"]).max().strftime("%Y-%m-%d %H:%M")
    st.sidebar.markdown(f"**Total Records:** {rows}")
    st.sidebar.markdown(f"**Last Update:** {last_time}")
except Exception:
    st.sidebar.markdown("**Dataset not loaded yet.**")

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<p style='text-align:center; color:#999; font-size:12px;'>Minimal Analytics Interface</p>",
    unsafe_allow_html=True
)

if upload_choice == "Upload Custom Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df_live = pd.read_csv(uploaded_file)
        else:
            df_live = pd.read_excel(uploaded_file)
        if "timestamp" in df_live.columns:
            df_live["timestamp"] = pd.to_datetime(df_live["timestamp"], errors="coerce")
        else:
            df_live["timestamp"] = pd.date_range(start=datetime.now(), periods=len(df_live), freq="5min")
    else:
        st.warning("📂 Please upload a dataset to continue.")
        st.stop()
else:
    @st.cache_data
    def load_live_data():
        return pd.read_csv("live_feed.csv", parse_dates=["timestamp"])
    df_live = load_live_data()

df = df_live.copy()
route_cat = {c: i for i, c in enumerate(meta["route_categories"])}
stop_cat = {c: i for i, c in enumerate(meta["stop_categories"])}
df["route_idx"] = df["route_id"].map(route_cat).fillna(-1).astype(int)
df["stop_idx"] = df["stop_id"].map(stop_cat).fillna(-1).astype(int)
df["hour"] = df["timestamp"].dt.hour
df["is_peak"] = df["hour"].isin([7,8,9,17,18,19]).astype(int)
df["route_stop_mean_delay"] = df.groupby(["route_id","stop_id"])["delay_min"].transform("mean").fillna(df["delay_min"].mean())
for f in meta["features"]:
    if f not in df.columns:
        df[f] = 0
preds = model.predict(df[meta["features"]])
df["predicted_delay_min"] = np.round(preds, 2)

avg_delay = df["predicted_delay_min"].mean()
busiest_route = df["route_id"].mode()[0]
avg_temp = df["weather_temp"].mean()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_delay:.2f} min</div><div class='metric-label'>Average Delay</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{busiest_route}</div><div class='metric-label'>Busiest Route</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_temp:.1f} °C</div><div class='metric-label'>Avg Temperature</div></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["▮ Predictions", "▮ Analytics", "▮ Heatmap", "▮ Export"])

with tab1:
    route = st.selectbox("Filter by Route", ["All"] + meta["route_categories"])
    stop = st.selectbox("Filter by Stop", ["All"] + meta["stop_categories"])
    df_filtered = df.copy()
    if route != "All":
        df_filtered = df_filtered[df_filtered["route_id"] == route]
    if stop != "All":
        df_filtered = df_filtered[df_filtered["stop_id"] == stop]
    st.dataframe(df_filtered[["timestamp","route_id","stop_id","predicted_delay_min"]].tail(50))
    st.line_chart(df_filtered[["timestamp","predicted_delay_min"]].set_index("timestamp"))

with tab2:
    st.subheader("Hourly Delay Trend")
    hourly_delay = df.groupby("hour")["predicted_delay_min"].mean().reset_index()
    fig = px.area(hourly_delay, x="hour", y="predicted_delay_min", title="", template="plotly_dark")
    fig.update_traces(line_color="white", fillcolor="rgba(255,255,255,0.1)")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", xaxis_title="Hour", yaxis_title="Avg Delay (min)")
    st.plotly_chart(fig, use_container_width=True)
    insights = []
    max_delay_route = df.groupby("route_id")["predicted_delay_min"].mean().idxmax()
    insights.append(f"🚌 Route **{max_delay_route}** shows the highest average delay.")
    if df["weather_precip"].mean() > 0.2:
        insights.append("🌧️ Rain conditions observed — potential delay impact.")
    if avg_delay > 5:
        insights.append("⚠️ Overall system delay is higher than normal.")
    for ins in insights:
        st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)

with tab3:
    st.subheader("Route-Stop Delay Heatmap")
    pivot = df.pivot_table(index="route_id", columns="stop_id", values="predicted_delay_min", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(pivot, cmap="gray", ax=ax)
    st.pyplot(fig)

with tab4:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions CSV", csv, "predicted_delays.csv", "text/csv")
