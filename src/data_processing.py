import pandas as pd
from datetime import time
import plotly.express as px


#Load NQ Futures Dataset
def data_processing(filename ):
    df = pd.read_csv(filename)

    #Convert Timestamp Column from text to Datetime format
    df["timestamp ET"] = pd.to_datetime(df["timestamp ET"])

    #Separate Date and Time Columns
    df["date"] = df["timestamp ET"].dt.date
    df["time"] = df["timestamp ET"].dt.time

    #Create the  5-Minute ORB =  Extract only the First Five one-minute candles in the opening range
    orb_5m = df[(df["time"] >= time(9,30))& (df["time"] <= time(9,34))]
    # print(orb_5m.head(10))

    #Create the Daily 5-Minute Opening Range and DATA VALIDATION
    rows_daily = orb_5m.groupby("date").size()

    #Show the Days with complete 5-minute opening ranges
    # print(rows_daily.value_counts())

    #Find the Dates with Missing Data
    # print(rows_daily[rows_daily!= 5])
    orb_daily = (orb_5m.groupby("date").agg(
        orb_high = ("high", "max"),
        orb_low = ("low", "min"),
        orb_close = ("close", "last"),
        orb_volume= ("volume", "sum"))
        .reset_index())
    orb_daily["orb_range"] = (orb_daily["orb_high"] - orb_daily["orb_low"])

    # print(orb_daily.head())
    # print(orb_daily["orb_range"].describe())

    valid_dates = rows_daily[rows_daily == 5].index
    orb_daily = orb_daily[orb_daily["date"].isin(valid_dates)].reset_index(drop=True)   #Renumber rows after DATA VALIDATION = 764 Trading Sessions


    #Create a Weekday Column to determine average ORB range by Weekday
    orb_daily["weekday"] = pd.to_datetime(orb_daily["date"]).dt.day_name()
    weekday_stats = (orb_daily.groupby("weekday")["orb_range"].mean().sort_values(ascending=False))
    # print(weekday_stats.round(2))  #Print the average ORB range per weekday
    # print(f"Smallest Average ORB: {weekday_stats.min():.2f} points")
    # print(f"Largest Average ORB: {weekday_stats.max():.2f} points")
    # print(f"Weekday with the Largest Average ORB: {weekday_stats.idxmax()}")
    # print(f"Weekday with the Smallest Average ORB: {weekday_stats.idxmin()}")



    #Date with the Largest ORB
    largest_orb = orb_daily.loc[orb_daily["orb_range"].idxmax()]
    # print(f"Date with the Largest ORB: {largest_orb["date"]}")
    # print(f"Weekday: {largest_orb["weekday"]}")
    # print(f"ORB Range: {largest_orb['orb_range']}")
    # print(f"ORB High: {largest_orb['orb_high']}")
    # print(f"ORB Low: {largest_orb['orb_low']}")
    # print(f"ORB Volume: {largest_orb['orb_volume']}")


    #Date with the Largest ORB Volume
    largest_volume = orb_daily.loc[orb_daily["orb_volume"].idxmax()]
    # print(f"Date with the Largest ORB Volume: {largest_volume["date"]}")
    # print(f"Weekday: {largest_volume["weekday"]}")
    # print(f"ORB Range: {largest_volume['orb_range']}")
    # print(f"ORB High: {largest_volume['orb_high']}")
    # print(f"ORB Low: {largest_volume['orb_low']}")
    # print(largest_volume)

    #Top 10 Largest ORB Ranges
    top10_orbs = (orb_daily.sort_values(by="orb_range", ascending=False).head(10))
    # print(f"Top 10 ORB Ranges:")
    # print(top10_orbs[["date","weekday","orb_range","orb_volume"]].to_string(index=False))

    #How many Trading Days were Mondays - Fridays
    weekday_counts = (orb_daily.groupby("weekday")["orb_range"].count())
    # print(weekday_counts)

    #Make Weekly Stats it Data Frame Not a Series
    weekday_stats =(orb_daily.groupby("weekday", as_index=False)["orb_range"].mean())      #as_index=False == makes the weekday a normal column instead of an index
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekday_stats["weekday"] = pd.Categorical(weekday_stats["weekday"], categories=weekday_order, ordered=True)
    weekday_stats = weekday_stats.sort_values(by="weekday")

    #Create Chart x= x-axis which is the Weekday  y= y-axis which is the ORB Range
    # fig = px.bar(weekday_stats, x="weekday", y="orb_range", title="Average ORB Range by Weekday", labels={"weekday":"Weekday", "orb_range":"Average ORB Range (in Points)"})
    # fig.update_traces(hovertemplate="<b>%{x}</b><br>" + "Average ORB: %{y:.2f} points")
    # fig.show()
    #

    #Create Histogram showing average size of ORB ranges and amount of times they happened
    # fig_history =px.histogram(orb_daily, x= "orb_range",  title="Frequencies of 5-Minute ORB Ranges" , labels={"orb_range": "ORB Range (Points)"})
    # fig_history.update_yaxes(title="Number of Days")
    # fig_history.update_traces(hovertemplate="ORB Range: %{x}<br>" + "Number of Days: %{y}")
    # fig_history.show()


    #Make Monthly Stats it Data Frame Not a Series
    #Create a Month Column to determine average ORB range by Month
    orb_daily["month"] = pd.to_datetime(orb_daily["date"]).dt.month_name() #Create Month column frm Date
    month_stats = (orb_daily.groupby(["month", "weekday"], as_index=False)["orb_range"].mean()) #Group by Month and Weekday
    month_stats["orb_range"] = month_stats["orb_range"].round(2)  #Calculate Average ORB
    month_order = ["January", "February", "March","April", "May", "June", "July", "August", "September", "October", "November", "December"]
    #Sort in Order (JAN --> DEC and MON --> FRI)
    month_stats["month"] = pd.Categorical(month_stats["month"], categories=month_order, ordered=True)
    month_stats["weekday"] = pd.Categorical(month_stats["weekday"], categories=weekday_order, ordered=True)
    month_stats = month_stats.sort_values(by=["month", "weekday"])   #Sort Everything
    # print(month_stats)

    # Create Heatmap
    heatmap = month_stats.pivot( index="month", columns="weekday", values="orb_range")
    # print(heatmap)
    fig_heatmap = px.imshow(heatmap, title="Average 5-Minute ORB Range by Month and Day", labels={"x": "Weekday", "y": "Month", "color": "Average ORB (Points)"}, text_auto=True)   #imshow Plots the Matrix as an image where the Values determine the Colors
    fig_heatmap.show()


    # print(type(weekday_stats))
    # print(weekday_stats)
    # print(weekday_stats.columns)


    #     orb_open = orb_5m.first(["open"])
    #
    # orb_range = orb_high - orb_low
    #
    # if orb_close > orb_open:
    #     orb_direction = "BULLISH"
    # elif orb_close < orb_open:
    #     orb_direction = "SHORT"
    # else:
    #     orb_direction = "NEUTRAL"