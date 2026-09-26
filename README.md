# NQ Futures ORB Price Action System

## Project Overview

This project was created with Python and Streamilt. It is an application that analyzes historical Nasdaq-100 (NQ) futures Opening Range Breakout (ORB) trades.

The application uses a historical one-minute NQ futures dataset to calculate ORB statistics, display interactive charts, and train a Random Forest Classifier model to predict whether a trade will result in a Win or Loss.
This system is designed as a decision support tool (DST) as it will not execute trades or connect to a brokerage account.

**Live App Link** https://nqorbprediction.streamlit.app/
**GitHub Repo** https://github.com/tashatrifecta/nq-orb-prediction-system/blob/master/app/app.py
**Author** Tasha D. Fitch 

## Technologies Used

* **Python 3** — Main programming language
* **Streamlit** — Web application interface
* **Pandas** — Data processing and feature engineering
* **scikit-learn** — Random Forest machine learning model
* **Plotly** — Interactive data visualizations
* **CSV** — Historical NQ futures dataset

## ORB Strategy

The Opening Range is created from the first five one-minute candles of the New York session:

-- ORB time: 09:30 AM - 09:34 AM ET
-- Trade window: 09:35 - 11:00 AM ET
-- Long Entry (Buys): Valid trade entry once the price breaks the ORB High and a one-minute candle closes above it.
-- Short Entry (Sells): Valid trade entry once the price breaks the ORB Low and a one-minute candle closes below it.

Risk to Reward Ratio: 2:1
Risking 0.5x of the ORB range to make 1.0x the ORB Range

--Take Profit (TP): 1.0x ORB Range
--Stop Loss (SL): 0.5x OrB Range

Historical trades are classified as Win, Loss, or Ambiguous


## Machine Learning

The project uses a Random Forest classification model imported from the scikit-learn library.
This model is trained using five model features, they are:

-- Orb range
-- ORB Volume
-- ORB Direction
-- Day of Week
-- 14-period ATR

The model is trained using an 80/20 split. The model trains on 80% of the historical data and is tested on the remaining 20% in chronological order.
** Only resolved Win and Loss trades are used to train and test the model.

## Project Files

```text
NQ_ORB_PriceActionSystem/
├── app/
│   └── app.py
├── data/
│   └── raw/
│       └── Dataset_NQ_1min_2022_2025.csv
├── src/
│   ├── data_processing.py
│   └── machine_learning.py
├── visuals/
│   └── charts.py
├── .gitignore
├── README.md
└── requirements.txt
```

## User Guide

1. Visit the NQ Futures ORB Prediction System website: https://nqorbprediction.streamlit.app/
2. Select a Month and Year from the filters on the left to analyze the historical period
3. Review the Historical ORB Summary for the selected period
4. Enter the current ORB Range, ORB Volume, ORB Direction, Day of Week, and ATR(14) in the Machine Learning Prediction section
5. Select Predict the Trade Outcome to receive a WIN or LOSS prediction and the probability of each outcome
6. Scroll to the ORB Visual Analysis section to review the historical charts and visualizations

User Guide(Local Setup)
1. Install Python 3
2. Download or clone the NQ Futures ORB Prediction System repository from GitHub
   git clone https://github.com/TashaTrifecta/NQ-ORB-Prediction-System
3. Open a terminal and navigate to the downloaded project folder
4. Install the required libraries using the requirements.txt file: python -m pip install -r requirements.txt
5. Confirm that Dataset_NQ_1min_2022_2025.csv is located in the data/raw folder
6. Start the application: python -m streamlit run app/app.py
7. Open the local Streamit address displayed in the terminal if the browser does not open automatically
8. Select a Month and Year from the filters on the left to analyze the historical period
9. Review the Historical ORB Summary for the selected period
10. Enter the current ORB Range, ORB Volume, ORB Direction, Day of Week, and ATR(14) in the Machine Learning Prediction section
11. Select Predict the Trade Outcome to receive a WIN or LOSS prediction and the probability of each outcome
12. Scroll to the ORB Visual Analysis section to review the historical charts and visualizations
