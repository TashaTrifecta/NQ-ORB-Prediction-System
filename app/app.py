import pandas as pd
import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


from src.data_processing import data_processing
from src.machine_learning import machine_learning
from visuals.charts import (create_weekday_chart,create_orb_histogram,create_heatmap,create_win_rate_chart)

st.set_page_config(page_title='NQ ORB Price Action System', page_icon="📈", layout="wide")

st.title("NQ Futures ORB Price Action System")

st.write("Interactive analysis and machine learning for the Nasdaq-100 Opening Range Breakout strategy")
st.sidebar.header("Filters")
year = st.sidebar.selectbox("Select Year", ["All", 2022 ,2023, 2024, 2025])
month = st.sidebar.selectbox("Select Month", ["All", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])

data_file = project_root / "data" / "raw" / "Dataset_NQ_1min_2022_2025.csv"
@st.cache_data
def load_data(filename):
    return data_processing(filename)

@st.cache_resource
def load_model(filename):
    return machine_learning(filename)

orb_daily = load_data(data_file)
rf_model = load_model(data_file)

orb_daily["date"] = pd.to_datetime(orb_daily["date"])
filtered_orb_daily = orb_daily.copy()
filtered_orb_daily["weekday"] = filtered_orb_daily["date"].dt.day_name()
filtered_orb_daily["month"] = filtered_orb_daily["date"].dt.month_name()

if year != "All":
    filtered_orb_daily = filtered_orb_daily[filtered_orb_daily["date"].dt.year == year]
if month != "All":
    filtered_orb_daily = filtered_orb_daily[filtered_orb_daily["date"].dt.month_name() == month]

#Calculate historical ORB summary statistics
st.subheader("Historical ORB Summary")
total_sessions = len(filtered_orb_daily)
total_trades = filtered_orb_daily["trade_outcome"].notna().sum()
resolved_trades = filtered_orb_daily[filtered_orb_daily["trade_outcome"].isin(["WIN", "LOSS"])]
total_resolved = len(resolved_trades)
total_wins = (filtered_orb_daily["trade_outcome"]== "WIN").sum()
total_losses = (filtered_orb_daily["trade_outcome"]== "LOSS").sum()

if total_resolved > 0:
    win_rate = float((total_wins / total_resolved) * 100)
else:
    win_rate = 0.0

average_orb_range = float(filtered_orb_daily["orb_range"].mean())
average_atr = float(filtered_orb_daily["atr_14"].mean())

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="ORB Sessions", value=total_sessions)
with col2:
    st.metric(label="Total Trades", value=total_trades)
with col3:
    st.metric(label="Total Wins", value=total_wins)
with col4:
    st.metric(label="Total Losses", value=total_losses)

col5, col6, col7 = st.columns(3)
with col5:
    st.metric(label="Win Rate", value=f"{win_rate:.1f}%")
with col6:
    st.metric(label="Average ORB Range", value=f"{average_orb_range:.1f} points")
with col7:
    st.metric(label="Average ATR(14)", value=f"{average_atr:.1f} points")

st.subheader("Machine Learning Prediction")
st.write("Enter the current ORB market conditions to estimate if the trade will reach the profit target before the stop loss")
#Create user input for ORB Range, volume, direction, day of week, and ATR
orb_range_input = st.number_input("ORB Range (Points)", min_value=0.0, value=50.0, step=0.5)
orb_volume_input = st.number_input("ORB Volume", min_value=0, value=10000, step=100)
orb_direction_input = st.selectbox("ORB Direction", ["BULLISH", "BEARISH"])
day_input = st.selectbox("Day of Week",["Monday","Tuesday","Wednesday","Thursday","Friday"])
atr_input = st.number_input("ATR(14)", min_value=0.0, value=15.0, step=0.1)
direction_encoded = {"BULLISH": 1, "BEARISH": 0}[orb_direction_input]
day_encoded = {"Monday": 0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4}[day_input]

#Create Prediction Button
prediction_input = pd.DataFrame([{"orb_range": orb_range_input, "orb_volume":orb_volume_input,"orb_direction":direction_encoded, "day_of_week": day_encoded,"atr_14":atr_input}])
if st.button("Predict the Trade Outcome"):
    if orb_range_input <= 0 or atr_input <= 0:
        st.warning("ORB Range and ATR(14) must be greater than zero")
    else:
        prediction = rf_model.predict(prediction_input)[0]
        probabilities = rf_model.predict_proba(prediction_input)[0]
        probability_map = dict(zip(rf_model.classes_, probabilities))
        win_probability = probability_map["WIN"]
        loss_probability = probability_map["LOSS"]
        st.write("Predicted Outcome:", prediction)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("WIN Probability", f"{win_probability:.1%}")
    with col2:
        st.metric("LOSS Probability", f"{loss_probability:.1%}")
    st.caption(
        "This prediction is based on historical ORB patterns and is intended "
        "for decision support only. It does not guarantee future trading outcomes."
    )

# Display model information for monitoring and documentation
with st.expander("System & Model Information"):
    st.write("Model Status: Loaded")
    st.write("Model Type: Random Forest Classification")
    st.write("Dataset Period: 2022–2025")
    st.write("Prediction Target: WIN / LOSS")
    st.write("Features Used: 5")
    st.write("Training Records: 552")
    st.write("Testing Records: 139")
    st.write("Testing Accuracy: 53.2%")

#Show Visualizations created in charts.py
st.subheader("ORB Visual Analysis")
win_rate_chart = create_win_rate_chart(filtered_orb_daily)
st.plotly_chart(win_rate_chart,  width="stretch")

heatmap = create_heatmap(filtered_orb_daily)
st.plotly_chart(heatmap, width="stretch")

weekday_chart = create_weekday_chart(filtered_orb_daily)
st.plotly_chart(weekday_chart,  width="stretch")

orb_histogram = create_orb_histogram(filtered_orb_daily)
st.plotly_chart(orb_histogram,  width="stretch")


