import pandas as pd
from datetime import time

#Load NQ Futures Dataset
def data_processing(filename):
    df = pd.read_csv(filename)

    #Convert Timestamp Column from text to Datetime format
    df["timestamp ET"] = pd.to_datetime(df["timestamp ET"])

    #Separate Date and Time Columns
    df["date"] = df["timestamp ET"].dt.date
    df["time"] = df["timestamp ET"].dt.time

    #Create the  5-Minute ORB =  Extract only the First Five one-minute candles in the opening range
    orb_5m = df[(df["time"] >= time(9,30))& (df["time"] <= time(9,34))]

    #Create the Daily 5-Minute Opening Range and DATA VALIDATION
    rows_daily = orb_5m.groupby("date").size()
    valid_dates = rows_daily[rows_daily == 5].index

    orb_daily = (orb_5m.groupby("date").agg(
        orb_open=("open", "first"),
        orb_high = ("high", "max"),
        orb_low = ("low", "min"),
        orb_close = ("close", "last"),
        orb_volume= ("volume", "sum"))
        .reset_index())
    post_orb = df[(df["time"] >= time(9,35))& (df["time"] <= time(10,00))]

    #Renumber rows after DATA VALIDATION = 764 Trading Sessions
    orb_daily = orb_daily[orb_daily["date"].isin(valid_dates)].reset_index(drop=True)
    orb_daily["orb_range"] = (orb_daily["orb_high"] - orb_daily["orb_low"])

    #Determine Bullish or Bearish Direction
    orb_daily["orb_direction"] = "NEUTRAL"
    orb_daily.loc[orb_daily["orb_close"] > orb_daily["orb_open"],"orb_direction"] = "BULLISH"
    orb_daily.loc[orb_daily["orb_close"] < orb_daily["orb_open"],"orb_direction"] = "BEARISH"

    #Create a Merge between the Daily ORB columns and Post ORB to compare the Pst orb High/Low agaisnt the 5M ORB High/Low
    post_orb_merged = post_orb.merge(orb_daily[["date", "orb_high", "orb_low", "orb_direction"]], on="date", how="left")
    post_orb_merged["high_break"] = (post_orb_merged["high"] >= post_orb_merged["orb_high"])
    post_orb_merged["low_break"] = (post_orb_merged["low"] <= post_orb_merged["orb_low"])

    #Determine what Side of the 5m ORB price broke first (High or Low or Ambiguous)
    first_high_break = (post_orb_merged[post_orb_merged["high_break"]].groupby("date")["time"].min())
    first_low_break = (post_orb_merged[post_orb_merged["low_break"]].groupby("date")["time"].min())

    #Create a table to compare both High and Low Breaks
    break_times = pd.concat([first_high_break, first_low_break], axis=1)
    break_times.columns = ["first_high_break", "first_low_break"]
    break_times["first_break"] = "NONE"
    break_times = break_times.reindex(orb_daily["date"])
    break_times["first_break"] = break_times["first_break"].fillna("NONE")
    #notna = There is a Break Time   isna = there is no Break Time avail
    break_times.loc[break_times["first_high_break"].notna() & break_times["first_low_break"].isna(), "first_break"] = "HIGH"  #High broke, Low did not
    break_times.loc[break_times["first_low_break"].notna() & break_times["first_high_break"].isna(), "first_break"] = "LOW"  #Low broke, High did not
    break_times.loc[break_times["first_high_break"].notna() & break_times["first_low_break"].notna() & (break_times["first_high_break"] < break_times["first_low_break"]), "first_break"] = "HIGH" #Both broke but High broke first
    break_times.loc[break_times["first_high_break"].notna() & break_times["first_low_break"].notna() & (break_times["first_low_break"] < break_times["first_high_break"]), "first_break"] = "LOW"  #Both broke but Low broke first
    break_times.loc[break_times["first_high_break"].notna() & break_times["first_low_break"].notna() &(break_times["first_high_break"] == break_times["first_low_break"]),"first_break"] = "AMBIGUOUS" #Both broke during the same 1-minute candle
    break_percentages = (break_times["first_break"].value_counts(normalize=True).mul(100).round(2))
    break_times = break_times.reset_index() #turns the index into a normal column

    #Merge Daily ORB with the Outcome of the first break times and ORB Features
    orb_daily_merged = orb_daily.merge(break_times[["date", "first_break"]], on="date", how="left")
    # print(orb_daily_merged[["date", "orb_direction", "orb_range", "orb_volume", "first_break"]].head(10))

    # print("ORB Trading Days:", len(orb_daily))
    # print("Break Time Days:", len(break_times))
    # print(break_times["first_break"].value_counts())
    # print(break_percentages.astype(str) + "%")

    # Create a Weekday Column to determine average ORB range by Weekday
    orb_daily["weekday"] = pd.to_datetime(orb_daily["date"]).dt.day_name()

    # Create a Month Column to determine average ORB range by Month
    orb_daily["month"] = pd.to_datetime(orb_daily["date"]).dt.month_name()  #Create Month column from Date

    # print(orb_daily[["date", "orb_open", "orb_close", "orb_direction"]].head(10))
    percentages = orb_daily["orb_direction"].value_counts(normalize=True).mul(100).round(2)
    # print(percentages.astype(str) + " %")
    # print(post_orb.head(30))

    # return orb_daily
    return orb_daily_merged
