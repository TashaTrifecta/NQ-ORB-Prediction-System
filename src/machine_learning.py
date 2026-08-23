import pandas as pd

from src.data_processing import data_processing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

def machine_learning(filename):
    ml_dataset = data_processing(filename)

    ml_dataset["day_of_week"] = pd.to_datetime(ml_dataset["date"]).dt.dayofweek
    ml_dataset = ml_dataset[ml_dataset["trade_outcome"].isin(["WIN","LOSS"])].reset_index(drop=True) #Keeps only W/L Outcomes
    ml_dataset = ml_dataset[ml_dataset["orb_direction"] != "NEUTRAL"].reset_index(drop=True) #Removes 1 neutral directional data to return zero missing feature values

    #Features and Target
    X = ml_dataset[["orb_range", "orb_volume", "orb_direction","day_of_week","atr_14"]].copy()        #X = Features
    X["orb_direction"] = X["orb_direction"].map({"BULLISH":1, "BEARISH":0})    #Maps the keywords BULLISH and BEARISH into numerical values for ML
    y = ml_dataset["trade_outcome"]                                              #Y = Target
    #ML Train and Test
    #Split Sessions into 80% Training and 20% Testing chronological split instead of randomizing because this is time-series market data
    split_index = int(len(y)*0.80)
    X_train = X.iloc[:split_index]   #Start at beginning and use everything before row 604
    X_test = X.iloc[split_index:]    #Start at row and use everything to the end
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    test_dates = ml_dataset["date"].iloc[split_index:].reset_index(drop=True)

    #Logistic Regression Model is a Classification model (it learns relationships between the orb features and whether high or low breaks first)
    #Create the model obj train it using these features and known answers
    #Previous Logistic Regression baseline model, Originally used to predict whether the ORB high or low would break first. Retained for development history but not used in the final WIN/LOSS model.
    # model = LogisticRegression()
    # model.fit(X_train, y_train)   #TRAIN THE MODEL USING HISTORICAL FEATURES AND THEIR KNOWN OUTCOMES

    #Test on unseen sessions
    # y_pred = model.predict(X_test)    #PREDICT H OR L FOR THE UNSEEN SESSIONS
    # y_probability = model.predict_proba(X_test)
    # print(y_probability[:10])
    # for i in range(10):
    #     print(f"Prediction: {y_pred[i]} | HIGH: {y_probability[i][0]:.2%} | LOW: {y_probability[i][1]:.2%}")

    #Measure accuracy which is correct predictions/all test predictions
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f" Model Accuracy: {accuracy:.2%}")
    # print(y_test.value_counts())
    # print(y_test.value_counts(normalize=True).mul(100).round(2))

    #Confusion Matrix
    #Compare actual outcomes against models predictions (diagonal values = correct predictions, off-diag values = incorrect predictions)
    # confus_matrix = confusion_matrix(y_test, y_pred, labels=["LOSS","WIN"])


    #Logistic Regression Coefficients (coeff show how each feature affects the logistic regression calculation)
    # print("Features:", X.columns)
    # print("Coefficients:", model.coef_)
    # print("Classes:", model.classes_)

    #Direction Prediction results  (combine each test session's date, actual outcome, prediction and model probabilities into one dataframe for analysis)
    #Logistic Regression and direction-only baseline both achieved 77.63% this suggests orb direction is the dominant feature in the current model
    # direction_pred = X_test["orb_direction"].map({0:"LOW", 1: "HIGH"})  #DIRECTION-ONLY BASELINE
    # direction_accuracy = accuracy_score(y_test, direction_pred)



    #ML results
    # results = pd.DataFrame({"date":test_dates, "actual":y_test.reset_index(drop=True), "predicted":y_pred, "high_probability":y_probability[:,0], "low_probability":y_probability[:,1]})
    # results["correct"] = results["actual"]==results["predicted"]
    # results["orb_range"] = X_test["orb_range"].reset_index(drop=True)
    # results["orb_volume"] = X_test["orb_volume"].reset_index(drop=True)
    # results["orb_direction"] = ml_dataset["orb_direction"].iloc[split_index:].reset_index(drop=True)
    # incorrect = results[results["correct"]==False]
    # direction_results = results.groupby("orb_direction")["correct"].agg(["count","sum","mean"])   #Groupby number of sessions, number that are correct and their accuracy
    # direction_results["accuracy"] = (direction_results["mean"] * 100).round(2)
    # # print(direction_results)
    # # print(results.groupby("correct")[["orb_range", "orb_volume"]].mean().round(2))
    #
    # #Year to year test of directional bias
    # ml_dataset["year"] = pd.to_datetime(ml_dataset["date"]).dt.year
    # ml_dataset["direction_correct"] = ((ml_dataset["orb_direction"] == "BULLISH") & (ml_dataset["first_break"] == "HIGH")) | ((ml_dataset["orb_direction"] == "BEARISH") & (ml_dataset["first_break"] == "LOW"))
    # yearly_direction = (ml_dataset.groupby(["year","orb_direction"])["direction_correct"].agg(["count","sum","mean"]))
    # yearly_direction["accuracy"] = (yearly_direction["mean"] * 100).round(2)
    # # print(yearly_direction)
    # yearly_accuracy = (ml_dataset.groupby("year")["direction_correct"].agg(["count", "sum", "mean"]))
    # yearly_accuracy["accuracy"] = (yearly_accuracy["mean"] * 100).round(2)
    # # print(yearly_accuracy)

    #Random Forrest classification model (use multiple decision trees and combines their decisions)
    #Random Forrest model achieved 100% training accuracy but only 68.42% testing accuracy(which indicates overfitting)
    #the model memorized patterns in the training data that did not generalize well to unseen sessions
    #the model ranked orb volume as the highest feature importance followed by orb range and orb direction
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=42)    #Build 200 Decision Trees 42 is a commonly used seed to make the random operations reproducible

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_train_pred = rf_model.predict(X_train)
    rf_train_accuracy = accuracy_score(y_train, rf_train_pred)

    # print("Random Forest Features: ",X.columns)
    # print("Feature Importances: ",rf_model.feature_importances_)
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


    # print(ml_dataset["trade_outcome"].value_counts())
    # print(ml_dataset.shape)
    # print("Outcome Counts:")
    # print(ml_dataset["trade_outcome"].value_counts(dropna=False))

    # print("\nActual Test Outcomes:")
    # print(y_test.value_counts())
    #
    # print("\nTest Outcome Percentages:")
    # print(y_test.value_counts(normalize=True).mul(100).round(2))

    # print("\nRF Test Prediction Counts:")
    # print(pd.Series(rf_pred).value_counts())
    #
    # print("\nRandom Forest Confusion Matrix:")
    # print(
    #     confusion_matrix(
    #         y_test,
    #         rf_pred,
    #         labels=["LOSS", "WIN"]
    #     )
    # )
    # print("\nRandom Forest Classification Report:")
    # print(
    #     classification_report(
    #         y_test,
    #         rf_pred,
    #         labels=["LOSS", "WIN"]
    #     )
    # )