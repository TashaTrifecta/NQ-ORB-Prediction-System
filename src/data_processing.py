import pandas as pd
from datetime import time


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

    # orb_daily.info()




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