import pandas as pd
import plotly.express as px

def create_weekday_chart(orb_daily):
    weekday_stats = (orb_daily.groupby("weekday", as_index=False)["orb_range"]).mean()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekday_stats["weekday"] = pd.Categorical(weekday_stats["weekday"], categories=weekday_order, ordered=True)
    weekday_stats = weekday_stats.sort_values(by="weekday")

    #Create Chart x= x-axis which is the Weekday  y= y-axis which is the ORB Range
    fig = px.bar(weekday_stats, x="weekday", y="orb_range", title="Average ORB Range by Weekday", labels={"weekday":"Weekday", "orb_range":"Average ORB Range (in Points)"}, color_discrete_sequence=["#636EFA"])
    fig.update_traces(hovertemplate="<b>%{x}</b><br>" + "Average ORB: %{y:.2f} points")
    return fig

    # Create Histogram showing average size of ORB ranges and amount of times they happened
def create_orb_histogram(orb_daily):
    fig_histo = px.histogram(orb_daily, x="orb_range", title="Frequencies of 5-Minute ORB Ranges",labels={"orb_range": "ORB Range (Points)"}, color_discrete_sequence=["#AB63FA"])
    fig_histo.update_yaxes(title="Number of Days")
    fig_histo.update_traces(hovertemplate="ORB Range: %{x}<br>" "Number of Days: %{y}")
    return fig_histo

    #Create Heatmap
def create_heatmap(orb_daily):
    month_stats = (orb_daily.groupby(["month", "weekday"], as_index=False)["orb_range"].mean())  # Group by Month and Weekday
    month_stats["orb_range"] = month_stats["orb_range"].round(2)  # Calculate Average ORB
    month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October","November", "December"]#Sort in Order (JAN --> DEC and MON --> FRI)
    month_stats["month"] = pd.Categorical(month_stats["month"], categories=month_order, ordered=True)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    month_stats["weekday"] = pd.Categorical(month_stats["weekday"], categories=weekday_order, ordered=True)
    month_stats = month_stats.sort_values(by=["month", "weekday"])  # Sort Everything
    heatmap = month_stats.pivot( index="month", columns="weekday", values="orb_range")

    # imshow Plots the Matrix as an image where the Values determine the Colors
    fig_heatmap = px.imshow(heatmap, title="Average 5-Minute ORB Range by Month and Day", labels={"x": "Weekday", "y": "Month", "color": "Average ORB (Points)"}, text_auto=True, aspect="auto", color_continuous_scale="Plasma")
    fig_heatmap.update_layout(height=700)
    return fig_heatmap