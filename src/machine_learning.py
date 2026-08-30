import pandas as pd

from src.data_processing import data_processing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

def machine_learning(filename):
    ml_dataset = data_processing(filename)

    #Keeps only WIN and LOSS outcomes
    ml_dataset["day_of_week"] = pd.to_datetime(ml_dataset["date"]).dt.dayofweek
    ml_dataset = ml_dataset[ml_dataset["trade_outcome"].isin(["WIN","LOSS"])].reset_index(drop=True)
    ml_dataset = ml_dataset[ml_dataset["orb_direction"] != "NEUTRAL"].reset_index(drop=True) #Removes 1 neutral directional data to return zero missing feature values

    #Features and Target
    X = ml_dataset[["orb_range", "orb_volume", "orb_direction","day_of_week","atr_14"]].copy()        #X = Features
    X["orb_direction"] = X["orb_direction"].map({"BULLISH":1, "BEARISH":0})    #Maps the keywords BULLISH and BEARISH into numerical values for ML
    y = ml_dataset["trade_outcome"]                                              #Y = Target

    #ML Train and Test
    #Split Sessions into 80% Training and 20% Testing chronological split instead of randomizing because this is time-series market data
    split_index = int(len(y)*0.80)
    X_train = X.iloc[:split_index]   #Start at the beginning
    X_test = X.iloc[split_index:]    #Start at split index and use everything to the end
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    #Random Forest Classification Model - Random Forest uses multiple decision trees and combines their predictions.
    #Balanced is used because the dataset has more LOSS trades than WIN trades
    #Without balancing, the model predicted LOSS most of the time  (class_weight = "balanced")
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
    # print(feature_importance)

    # print(f"Random Forest Accuracy: {rf_accuracy:.2%}")
    # print(f"Random Forest Training Accuracy: {rf_train_accuracy:.2%}")
    # print(f"Random Forest Testing Accuracy: {rf_accuracy:.2%}")
    # print("RF Train Correct:", (y_train.to_numpy() == rf_train_pred).sum())
    # print("RF Train Total:", len(y_train))
    # print("X Train:", X_train.shape)
    # print("RF Train Predictions:", len(rf_train_pred))

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

    # Shows where the model was right and wrong:[[correct LOSS, LOSS predicted as WIN],[WIN predicted as LOSS, correct WIN]]
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

    return rf_model