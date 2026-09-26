import pandas as pd
from datetime import time

#Load NQ Futures Dataset
def data_processing(filename):
    df = pd.read_csv(filename)

    #Convert Timestamp Column from text to Datetime format
    df["timestamp ET"] = pd.to_datetime(df["timestamp ET"])
    #Calculate 14 period ATR
    df["prev_close"] = df["close"].shift(1)
    df["true_range"] = pd.concat([df["high"] - df["low"],(df["high"] - df["prev_close"]).abs(),(df["low"] - df["prev_close"]).abs()],axis=1).max(axis=1)
    df["atr_14"] = (df["true_range"].rolling(window=14).mean())

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
    trade_window = df[(df["time"] >= time(9,35))& (df["time"] <= time(11,00))]

    #Renumber rows after DATA VALIDATION = 764 Trading Sessions
    orb_daily = orb_daily[orb_daily["date"].isin(valid_dates)].reset_index(drop=True)
    orb_daily["orb_range"] = (orb_daily["orb_high"] - orb_daily["orb_low"])

    #Determine Bullish or Bearish Direction
    orb_daily["orb_direction"] = "NEUTRAL"
    orb_daily.loc[orb_daily["orb_close"] > orb_daily["orb_open"],"orb_direction"] = "BULLISH"
    orb_daily.loc[orb_daily["orb_close"] < orb_daily["orb_open"],"orb_direction"] = "BEARISH"

    #Create a Merge between the Daily ORB columns and Post ORB to compare the Post orb High/Low agaisnt the 5M ORB High/Low
    post_orb_merged = trade_window.merge(orb_daily[["date", "orb_high", "orb_low", "orb_range", "orb_direction"]], on="date", how="left")
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
    break_times = break_times.reset_index() #turns the index into a normal column

    #Merge Daily ORB with the Outcome of the first break times
    orb_daily_merged = orb_daily.merge(break_times[["date", "first_break"]], on="date", how="left")
    post_orb_merged = post_orb_merged.merge(break_times[["date", "first_break"]],on="date",how="left")
    # print(orb_daily_merged[["date", "orb_direction", "orb_range", "orb_volume", "first_break"]].head(10))
    post_orb_merged["long_entry_signal"] = (post_orb_merged["close"] > post_orb_merged["orb_high"])
    post_orb_merged["short_entry_signal"] = (post_orb_merged["close"] < post_orb_merged["orb_low"])

    #Determine Valid Entries
    #High Broke 1st and candle closed above orb high or low broke 1st and candle closed below orb low
    post_orb_merged["valid_entry_signal"] = (((post_orb_merged["first_break"] == "HIGH") & (post_orb_merged["long_entry_signal"])) | ((post_orb_merged["first_break"] == "LOW") & (post_orb_merged["short_entry_signal"])))
    valid_entries = post_orb_merged[post_orb_merged["valid_entry_signal"]].copy()
    first_entries = (valid_entries.sort_values(["date", "time"]).groupby("date").first().reset_index())
    # print(first_entries[["date", "time", "first_break", "close", "orb_high", "orb_low", "orb_range"]].head(10))
    # print("Sessions with Entry: ", len(first_entries))

    #Calculate Take profit and Stop loss prices (TP and SL)
    tp_multiplier = 1.0
    sl_multiplier = 0.5
    #Create Entry Price and LONG and SHORT LEVELS
    first_entries["entry_price"] = first_entries["close"]
    first_entries["take_profit"] = 0.0
    first_entries["stop_loss"] = 0.0

    #Long(Buys)
    first_entries.loc[first_entries["first_break"] == "HIGH", "take_profit"] = (first_entries["entry_price"] + first_entries["orb_range"] * tp_multiplier)
    first_entries.loc[first_entries["first_break"] == "HIGH", "stop_loss"] = (first_entries["entry_price"] - first_entries["orb_range"] * sl_multiplier)

    #Shorts(Sells)
    first_entries.loc[first_entries["first_break"] == "LOW", "take_profit"] = (first_entries["entry_price"] - first_entries["orb_range"] * tp_multiplier)
    first_entries.loc[first_entries["first_break"] == "LOW", "stop_loss"] = (first_entries["entry_price"] + first_entries["orb_range"] * sl_multiplier)
    first_entries = first_entries.rename(columns={"time": "entry_time"})
    trade_candles = trade_window.merge(first_entries[["date", "entry_time", "first_break", "entry_price", "take_profit", "stop_loss"]], on="date", how="inner")
    trade_candles = trade_candles[trade_candles["time"] > trade_candles["entry_time"]].copy()
    # print(first_entries[["date","first_break", "orb_range", "entry_price", "take_profit", "stop_loss"]].head(10))

    #Create Valid Entry Candles and their TP and SL Price for LONGS and SHORTS
    trade_candles["tp_hit"] = False
    trade_candles["sl_hit"] = False
    trade_candles.loc[trade_candles["first_break"]== "HIGH", "tp_hit"] = (trade_candles["high"] >= trade_candles["take_profit"])
    trade_candles.loc[trade_candles["first_break"]== "HIGH", "sl_hit"] = (trade_candles["low"] <= trade_candles["stop_loss"])
    trade_candles.loc[trade_candles["first_break"]== "LOW", "tp_hit"] = (trade_candles["low"] <= trade_candles["take_profit"])
    trade_candles.loc[trade_candles["first_break"]== "LOW", "sl_hit"] = (trade_candles["high"] >= trade_candles["stop_loss"])

    #Create the candles for the First candle to Hit TP and SL
    first_tp_hit = (trade_candles[trade_candles["tp_hit"]].groupby("date")["time"].min())
    first_sl_hit = (trade_candles[trade_candles["sl_hit"]].groupby("date")["time"].min())
    # print(f"First TP HIT:\n{first_tp_hit.head(10)}")
    # print(f"First SL HIT:\n{first_sl_hit.head(10)}")

    #Trade Outcome Table
    outcome_times = pd.concat([first_tp_hit, first_sl_hit], axis=1)
    outcome_times.columns = ["first_tp_hit", "first_sl_hit"]
    outcome_times["trade_outcome"] = "NO_RESULT"

    #TP HIT, SL did not
    outcome_times.loc[outcome_times["first_tp_hit"].notna() & outcome_times["first_sl_hit"].isna(),"trade_outcome"] = "WIN"
    # SL HIT, TP did not
    outcome_times.loc[outcome_times["first_sl_hit"].notna() & outcome_times["first_tp_hit"].isna(),"trade_outcome"] = "LOSS"
    #Both TP and SL HIT
    outcome_times.loc[outcome_times["first_tp_hit"].notna() & outcome_times["first_sl_hit"].notna() & (outcome_times["first_tp_hit"] < outcome_times["first_sl_hit"]),"trade_outcome"] = "WIN"   #TP 1ST
    outcome_times.loc[outcome_times["first_tp_hit"].notna() & outcome_times["first_sl_hit"].notna() & (outcome_times["first_sl_hit"] < outcome_times["first_tp_hit"]),"trade_outcome"] = "LOSS"  #SL 1ST
    outcome_times.loc[outcome_times["first_tp_hit"].notna() & outcome_times["first_sl_hit"].notna() & (outcome_times["first_tp_hit"] == outcome_times["first_sl_hit"]), "trade_outcome"] = "AMBIGUOUS"  # TP 1ST
    outcome_times = outcome_times.reindex(first_entries["date"])
    outcome_times["trade_outcome"] = outcome_times["trade_outcome"].fillna("NO_RESULT")
    outcome_times = outcome_times.reset_index()
    orb_daily_merged = orb_daily_merged.merge(outcome_times[["date", "trade_outcome"]], on="date", how="left")
    # Merge ATR at trade entry into daily ORB dataset
    orb_daily_merged = orb_daily_merged.merge(first_entries[["date", "atr_14"]],on="date",how="left")

    #Data Validation: total entries should match total outcomes
    # print("Total Entries: ", len(first_entries))
    # print("Total Outcomes: ", len(outcome_times))

    #Return processed daily ORB dataset
    return orb_daily_merged
