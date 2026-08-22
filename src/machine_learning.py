from src.data_processing import data_processing

def machine_learning(filename):
    ml_dataset = data_processing(filename)
    ml_dataset = ml_dataset[ml_dataset["first_break"].isin(["HIGH","LOW"])].reset_index(drop=True) #KEEPS ONLY H/L OUTCOMES
    ml_dataset = ml_dataset[ml_dataset["orb_direction"] != "NEUTRAL"].reset_index(drop=True) #REMOVES 1 NEUTRAL DIRECTIONAL DATA TO RETURN ZERO MISSING FEATURE VALUES

    X = ml_dataset[["orb_range", "orb_volume", "orb_direction"]] #X = FEATURES
    X["orb_direction"] = X["orb_direction"].map({"BULLISH":1, "BEARISH":0}) #MAPS THE KEYWORDS BULLISH AND BEARISH INTO 0 OR 1 FOR READABLE MACHINE LANG

    y = ml_dataset["first_break"]  #Y = TARGET

    #SPLIT SESSIONS INTO 80% TRAINING AND 20% TESTING
    split_index = int(len(y)*0.80)
    x_train = X.iloc[:split_index]   #START AT BEGINNING AND USE EVERYTHING BEFORE ROW 604
    x_test = X.iloc[split_index:]    #START AT ROW AND USE EVERYTHING TO THE END
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("X Train:", x_train.shape)
    print("X Test:", x_test.shape)
    print("Y Train:", y_train.shape)
    print("Y Test:", y_test.shape)
