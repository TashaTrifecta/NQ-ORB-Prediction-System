import pandas as pd

#Load NQ Futures Dataset
def data_processing(filename ):
    df = pd.read_csv(filename)
    df["timestamp ET"] = pd.to_datetime(df["timestamp ET"])
    df.info()
    print(df["timestamp ET"][0])