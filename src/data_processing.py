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
    orb_daily = (orb_5m.groupby("date").agg(
        orb_high = ("high", "max"),
        orb_low = ("low", "min"),
        orb_close = ("close", "last"),
        orb_volume= ("volume", "sum"))
        .reset_index())
    orb_daily["orb_range"] = (orb_daily["orb_high"] - orb_daily["orb_low"])
    valid_dates = rows_daily[rows_daily == 5].index
    orb_daily = orb_daily[orb_daily["date"].isin(valid_dates)].reset_index(drop=True)   #Renumber rows after DATA VALIDATION = 764 Trading Sessions
    # Create a Weekday Column to determine average ORB range by Weekday
    orb_daily["weekday"] = pd.to_datetime(orb_daily["date"]).dt.day_name()

    # Create a Month Column to determine average ORB range by Month
    orb_daily["month"] = pd.to_datetime(orb_daily["date"]).dt.month_name()  #Create Month column from Date

    return orb_daily

