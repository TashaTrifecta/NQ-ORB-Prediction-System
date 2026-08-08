import pandas as pd
import streamlit as sl
from src.data_processing import data_processing
from visuals.charts import create_weekday_chart
from visuals.charts import create_orb_histogram
from visuals.charts import create_heatmap

orb_daily = data_processing("data/raw/Dataset_NQ_1min_2022_2025.csv")

sl.sidebar.header("Filters")
orb_daily["date"] = pd.to_datetime(orb_daily["date"])

years = sorted(orb_daily["date"].dt.year.unique())
selected_year = sl.sidebar.selectbox("Select Year", options=["All"] + years)

month_order = ["January", "February", "March", "April","May", "June", "July", "August","September", "October", "November", "December"]
selected_month = sl.sidebar.selectbox("Select Month", options=["All"] + month_order)

#Start with all data
filtered_orb_daily = orb_daily

#Filter by year if a year is selected
if selected_year != "All":
    filtered_orb_daily = filtered_orb_daily[filtered_orb_daily["date"].dt.year == selected_year]

#Filter by month if a month is selected
if selected_month != "All":
    filtered_orb_daily = filtered_orb_daily[filtered_orb_daily["month"] == selected_month]

sl.title("NQ Price Action System")
sl.write("5-Minute Opening Range Breakout Analytics Dashboard")

average_orb = filtered_orb_daily["orb_range"].mean()
largest_orb = filtered_orb_daily["orb_range"].max()
trading_days = len(filtered_orb_daily)

col1, col2, col3 = sl.columns(3)
with col1:
    sl.metric(label="Average ORB Range", value=f"{average_orb:.2f} points")
with col2:
    sl.metric(label="Largest ORB Range", value=f"{largest_orb:.2f} points")
with col3:
    sl.metric(label="Trading Days", value=f"{trading_days} days")

weekday_chart = create_weekday_chart(filtered_orb_daily)
sl.plotly_chart(weekday_chart, use_container_width=True)

histogram = create_orb_histogram(filtered_orb_daily)
sl.plotly_chart(histogram, use_container_width=True)

heatmap = create_heatmap(filtered_orb_daily)
sl.plotly_chart(heatmap, use_container_width=True)
