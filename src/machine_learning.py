from statistics import mean

import pandas as pd
from src.data_processing import data_processing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def machine_learning(filename):
    ml_dataset = data_processing(filename)
    ml_dataset = ml_dataset[ml_dataset["first_break"].isin(["HIGH","LOW"])].reset_index(drop=True) #KEEPS ONLY H/L OUTCOMES
    ml_dataset = ml_dataset[ml_dataset["orb_direction"] != "NEUTRAL"].reset_index(drop=True) #REMOVES 1 NEUTRAL DIRECTIONAL DATA TO RETURN ZERO MISSING FEATURE VALUES

    #FEATURES AND TARGET
    X = ml_dataset[["orb_range", "orb_volume", "orb_direction"]] #X = FEATURES
    X["orb_direction"] = X["orb_direction"].map({"BULLISH":1, "BEARISH":0}) #MAPS THE KEYWORDS BULLISH AND BEARISH INTO NUMERICAL VALUES FOR ML
    y = ml_dataset["first_break"]  #Y = TARGET

    #ML TRAIN AND TEST
    #SPLIT SESSIONS INTO 80% TRAINING AND 20% TESTING CHRONOLOGICAL SPLIT INSTEAD OF RANDOMIZING BECAUSE THIS IS TIME-SERIES MARKET DATA
    split_index = int(len(y)*0.80)
    X_train = X.iloc[:split_index]   #START AT BEGINNING AND USE EVERYTHING BEFORE ROW 604
    X_test = X.iloc[split_index:]    #START AT ROW AND USE EVERYTHING TO THE END
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    test_dates = ml_dataset["date"].iloc[split_index:].reset_index(drop=True)

    #LOGISTIC REGRESSION MODEL IS A CLASSIFICATION MODEL (IT LEARNS RELATIONSHIPS BETWEEN THE ORB FEATURES AND WHETHER HIGH OR LOW BREAKS FIRST)
    #CREATE THE MODEL OBJ TRAIN IT USING THESE FEATURES AND KNOWN ANSWERS
    model = LogisticRegression()
    model.fit(X_train, y_train)   #TRAIN THE MODEL USING HISTORICAL FEATURES AND THEIR KNOWN OUTCOMES

    #TEST ON UNSEEN SESSIONS
    y_pred = model.predict(X_test)    #PREDICT H OR L FOR THE UNSEEN SESSIONS
    y_probability = model.predict_proba(X_test)
    # print(y_probability[:10])
    # for i in range(10):
    #     print(f"Prediction: {y_pred[i]} | HIGH: {y_probability[i][0]:.2%} | LOW: {y_probability[i][1]:.2%}")

    #MEASURE ACCURACY WHICH IS CORRECT PREDICTIONS/ALL TEST PREDICTIONS
    accuracy = accuracy_score(y_test, y_pred)
    # print(f" Model Accuracy: {accuracy:.2%}")
    # print(y_test.value_counts())
    # print(y_test.value_counts(normalize=True).mul(100).round(2))

    #CONFUSION MATRIX
    #COMPARE ACTUAL OUTCOMES AGAINST MODELS PREDICTIONS (DIAGONAL VALUES = CORRECT PREDICTIONS, OFF-DIAG VALUES = INCORRECT PREDICTIONS)
    confus_matrix = confusion_matrix(y_test, y_pred, labels=["LOW","HIGH"])


    #LOGISTIC REGRESSION COEFFICIENTS (COEFF SHOW HOW EACH FEATURE AFFECTS THE LOGISTIC REGRESSION CALCULATION)
    # print("Features:", X.columns)
    # print("Coefficients:", model.coef_)
    # print("Classes:", model.classes_)

    #DIRECTION PREDICTION RESULTS  (COMBINE EACH TEST SESSION'S DATE, ACTUAL OUTCOME, PREDICTION AND MODEL PROBABILITIES INTO ONE DATAFRAME FOR ANALYSIS)
    #LOGISTIC REGRESSION AND DIRECTION-ONLY BASELINE BOTH ACHIEVED 77.63% THIS SUGGESTS ORB DIRECTION IS THE DOMINANT FEATURE IN THE CURRENT MODEL
    direction_pred = X_test["orb_direction"].map({0:"LOW", 1: "HIGH"})  #DIRECTION-ONLY BASELINE
    direction_accuracy = accuracy_score(y_test, direction_pred)



    #ML RESULTS
    results = pd.DataFrame({"date":test_dates, "actual":y_test.reset_index(drop=True), "predicted":y_pred, "high_probability":y_probability[:,0], "low_probability":y_probability[:,1]})
    results["correct"] = results["actual"]==results["predicted"]
    results["orb_range"] = X_test["orb_range"].reset_index(drop=True)
    results["orb_volume"] = X_test["orb_volume"].reset_index(drop=True)
    results["orb_direction"] = ml_dataset["orb_direction"].iloc[split_index:].reset_index(drop=True)
    incorrect = results[results["correct"]==False]
    direction_results = results.groupby("orb_direction")["correct"].agg(["count","sum","mean"])   #GROUPBY NUMBER OF SESSIONS, NUMBER THAT ARE CORRECT AND THEIR ACCURACY
    direction_results["accuracy"] = (direction_results["mean"] * 100).round(2)
    # print(direction_results)
    # print(results.groupby("correct")[["orb_range", "orb_volume"]].mean().round(2))

    #YEAR TO YEAR TEST OF DIRECTIONAL BIAS
    ml_dataset["year"] = pd.to_datetime(ml_dataset["date"]).dt.year
    ml_dataset["direction_correct"] = ((ml_dataset["orb_direction"] == "BULLISH") & (ml_dataset["first_break"] == "HIGH")) | ((ml_dataset["orb_direction"] == "BEARISH") & (ml_dataset["first_break"] == "LOW"))
    yearly_direction = (ml_dataset.groupby(["year","orb_direction"])["direction_correct"].agg(["count","sum","mean"]))
    yearly_direction["accuracy"] = (yearly_direction["mean"] * 100).round(2)
    # print(yearly_direction)
    yearly_accuracy = (ml_dataset.groupby("year")["direction_correct"].agg(["count", "sum", "mean"]))
    yearly_accuracy["accuracy"] = (yearly_accuracy["mean"] * 100).round(2)
    # print(yearly_accuracy)

    #RANDOM FORREST CLASSIFICATION MODEL (USE MULTIPLE DECISION TREES AND COMBINES THEIR DECISIONS)
    #RANDOM FORREST MODEL ACHIEVED 100% TRAINING ACCURACY BUT ONLY 68.42% TESTING ACCURACY(WHICH INDICATES OVERFITTING, THE MODEL MEMORIZED PATTERNS IN THE TRAINING DATA
    #THE MODEL RANKED ORB VOLUME AS THE HIGHEST FEATURE IMPORTANCE FOLLOWED BY ORB RANGE AND ORB DIRECTION
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42)    #BUILD 100 DECISION TREES 42 IS A COMMONLY USED SEED TO MAKE THE RANDOM OPERATIONS REPRODUCIBLE

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_train_pred = rf_model.predict(X_train)
    rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
    rf_model.feature_importances_
    print("Random Forest Features: ",X.columns)
    print("Feature Importances: ",rf_model.feature_importances_)
    feature_importance = pd.DataFrame({"feature": X.columns, "importance":rf_model.feature_importances_})
    feature_importance = feature_importance.sort_values("importance", ascending=False)
    print(feature_importance)

    # print(f"Random Forest Accuracy: {rf_accuracy:.2%}")
    # print(f"Random Forest Training Accuracy: {rf_train_accuracy:.2%}")
    # print(f"Random Forest Testing Accuracy: {rf_accuracy:.2%}")
    # print("RF Train Correct:", (y_train.to_numpy() == rf_train_pred).sum())
    # print("RF Train Total:", len(y_train))
    # print("X Train:", X_train.shape)
    # print("RF Train Predictions:", len(rf_train_pred))

    # print("X Test:", X_test.shape)
    # print("RF Test Predictions:", len(rf_pred))

    # print("Y Train:", y_train.shape)
    # print("Y Test:", y_test.shape)

    # print(rf_model)
    # print(rf_model.get_params())
    # print("Train Prediction Counts:")
    # print(pd.Series(rf_train_pred).value_counts())

    # print("Train Actual Counts:")
    # print(y_train.value_counts())

    # print("Test Prediction Counts:")
    # print(pd.Series(rf_pred).value_counts())

    # print("First 10 RF Train Predictions:", rf_train_pred[:10])
    # print("First 10 Actual Train Values:", y_train.iloc[:10].values)
