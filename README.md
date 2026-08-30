# NQ Futures ORB Price Action System

## 1. Project Overview

The NQ Futures ORB Price Action System is an interactive data analysis and
machine learning application designed to analyze historical Nasdaq-100 (NQ)
futures Opening Range Breakout (ORB) trading patterns.

The application processes historical one-minute NQ futures data, calculates
ORB market statistics, displays interactive visualizations, and uses a Random
Forest classification model to estimate whether an ORB trade will reach its
profit target before its stop loss.

The system is intended to provide historical market analysis and decision
support. It does not execute trades or guarantee future trading results.


## 2. Features

The application includes the following features:

- Historical analysis of NQ Opening Range Breakout trading sessions
- Calculation of the five-minute Opening Range
- Identification of bullish and bearish ORB trade entries
- Calculation of profit target and stop-loss levels
- Classification of historical trades as WIN, LOSS, or NO_RESULT
- Calculation of 14-period Average True Range (ATR)
- Interactive filtering by year and month
- Historical ORB summary statistics
- ORB win-rate analysis
- Average ORB range analysis
- Interactive Plotly visualizations
- Random Forest machine learning classification
- User-entered market conditions for trade outcome predictions
- WIN and LOSS probability estimates


## 3. ORB Trading Methodology

The Opening Range Breakout is calculated using the first five minutes of the
regular U.S. stock market session.

### Opening Range

- Opening Range: 9:30 AM through 9:34 AM Eastern Time
- ORB High: Highest price during the five-minute opening range
- ORB Low: Lowest price during the five-minute opening range
- ORB Range: ORB High minus ORB Low

### Trade Window

Potential trade entries are evaluated from:

- 9:35 AM through 11:00 AM Eastern Time

The system identifies whether price first breaks the ORB High or ORB Low.

A long entry occurs when the first break is above the ORB High and a candle
closes above the ORB High.

A short entry occurs when the first break is below the ORB Low and a candle
closes below the ORB Low.

Sessions in which both the ORB High and ORB Low are touched on the same
one-minute candle are classified as ambiguous and are not used as valid trade
entries.

### Profit Target and Stop Loss

The strategy uses a 2:1 reward-to-risk ratio.

- Profit Target: 1.0 times the ORB range
- Stop Loss: 0.5 times the ORB range

For long trades:

- Profit Target = Entry Price + ORB Range
- Stop Loss = Entry Price - (0.5 × ORB Range)

For short trades:

- Profit Target = Entry Price - ORB Range
- Stop Loss = Entry Price + (0.5 × ORB Range)

A trade is classified as a WIN when the profit target is reached before the
stop loss.

A trade is classified as a LOSS when the stop loss is reached before the
profit target.

A trade is classified as NO_RESULT when neither the profit target nor the
stop loss is reached during the defined trade evaluation period.


## 4. Dataset

The application uses historical one-minute Nasdaq-100 futures market data.

The dataset contains market information including:

- Timestamp
- Open price
- High price
- Low price
- Close price
- Volume

The raw dataset is stored in:

```text
data/raw/
```

The primary dataset used by the application is:

```text
Dataset_NQ_1min_2022_2025.csv
```

The application processes the raw market data to create the variables needed
for ORB analysis and machine learning.

Processed features include:

- ORB High
- ORB Low
- ORB Range
- ORB Volume
- ORB Direction
- Day of Week
- ATR(14)
- Trade Entry
- Profit Target
- Stop Loss
- Trade Outcome


## 5. Data Processing

Data preparation and feature engineering are performed in
`src/data_processing.py`.

The data-processing process includes:

1. Loading the historical CSV dataset.
2. Converting timestamps into usable date and time values.
3. Calculating the 14-period Average True Range.
4. Identifying the five-minute Opening Range.
5. Calculating the ORB High, ORB Low, range, and volume.
6. Determining ORB direction.
7. Identifying the first ORB boundary break.
8. Identifying valid trade entries.
9. Calculating profit target and stop-loss prices.
10. Evaluating whether the profit target or stop loss was reached first.
11. Assigning the resulting trade outcome.
12. Merging the calculated features into the daily ORB dataset.

Pandas is used for data parsing, cleaning, filtering, grouping, aggregation,
feature engineering, and dataset preparation.


## 6. Machine Learning Model

The predictive component of the application uses a Random Forest
classification model implemented with scikit-learn.

The model predicts one of two resolved trade outcomes:

- WIN
- LOSS

### Model Features

The Random Forest uses five input features:

1. ORB Range
2. ORB Volume
3. ORB Direction
4. Day of Week
5. ATR(14)

ORB Direction is encoded as:

- BULLISH = 1
- BEARISH = 0

Day of Week is encoded as:

- Monday = 0
- Tuesday = 1
- Wednesday = 2
- Thursday = 3
- Friday = 4

Only resolved WIN and LOSS trades with a valid bullish or bearish ORB
direction are included in the machine learning dataset.

### Training and Testing

The dataset is divided chronologically:

- 80% training data
- 20% testing data

A chronological split is used so that earlier historical observations are
used for training and later observations are used for testing.

The Random Forest model uses class balancing because the historical dataset
contains more losing trades than winning trades.

Model evaluation includes:

- Training accuracy
- Testing accuracy
- Confusion matrix
- Precision
- Recall
- F1-score
- Feature importance

The model is intended for decision support and should not be interpreted as a
guarantee of future market performance.


## 7. Interactive Dashboard

The user interface is developed with Streamlit.

The dashboard provides historical ORB statistics including:

- ORB Sessions
- Total Trades
- Total Wins
- Total Losses
- Win Rate
- Average ORB Range
- Average ATR(14)

Users can interactively filter historical results by:

- Year
- Month

The machine learning section allows users to enter:

- ORB Range
- ORB Volume
- ORB Direction
- Day of Week
- ATR(14)

The application then returns:

- Predicted trade outcome
- WIN probability
- LOSS probability


## 8. Data Visualizations

Plotly is used to create interactive visualizations for data exploration and
analysis.

The dashboard includes multiple visualization types:

### ORB Win Rate by Weekday

A bar chart displays the historical percentage of resolved ORB trades that
resulted in a WIN for each weekday.

### Average ORB Range Heatmap

A heatmap displays average ORB ranges across months and weekdays.

### Average ORB Range by Weekday

A bar chart compares average five-minute ORB ranges by weekday.

### ORB Range Distribution

A histogram displays the distribution and frequency of historical ORB range
sizes.

The visualizations update when the user changes the year or month filters.


## 9. Project Structure

```text
NQ_ORB_PriceActionSystem/
│
├── app/
│   └── app.py
│
├── data/
│   ├── processed/
│   └── raw/
│       └── Dataset_NQ_1min_2022_2025.csv
│
├── docs/
│
├── src/
│   ├── data_processing.py
│   └── machine_learning.py
│
├── visuals/
│   └── charts.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Main Components

`src/data_processing.py`

Processes the historical dataset, calculates ORB statistics, identifies
trades, determines trade outcomes, and creates features used by the machine
learning model.

`src/machine_learning.py`

Prepares the machine learning dataset, trains the Random Forest classifier,
and evaluates model performance.

`visuals/charts.py`

Creates the interactive Plotly charts used by the Streamlit dashboard.

`app/app.py`

Provides the Streamlit user interface, interactive filters, historical
summary statistics, machine learning prediction inputs, and visualizations.


## 10. Requirements

The application requires Python 3 and the Python packages listed in
`requirements.txt`.

Primary libraries used by the project include:

- pandas
- NumPy
- Matplotlib
- Plotly
- scikit-learn
- Streamlit

The dependency versions are specified in `requirements.txt` to provide a
consistent and reproducible development environment.

Install the required dependencies from the project root directory:

```powershell
python -m pip install -r requirements.txt
```


## 11. Dataset Setup

The application uses historical one-minute Nasdaq-100 (NQ) futures data.

Place the dataset in the following directory:

```text
data/raw/
```

The expected dataset filename is:

```text
Dataset_NQ_1min_2022_2025.csv
```

The complete file path should be:

```text
data/raw/Dataset_NQ_1min_2022_2025.csv
```


## 12. Running the Application

From the root directory of the project, install the required dependencies:

```powershell
python -m pip install -r requirements.txt
```

Launch the Streamlit application:

```powershell
python -m streamlit run app/app.py
```

Streamlit will start the application and provide a local address that can be
opened in a web browser.


## 13. Using the Application

1. Launch the Streamlit application.
2. Use the sidebar filters to select a year and month for historical analysis.
3. Review the Historical ORB Summary, including total sessions, trades, wins,
   losses, win rate, average ORB range, and average ATR(14).
4. Enter the current ORB market conditions in the Machine Learning Prediction
   section:
   - ORB Range
   - ORB Volume
   - ORB Direction
   - Day of Week
   - ATR(14)
5. Select **Predict the Trade Outcome**.
6. Review the predicted WIN or LOSS outcome and the associated probabilities.
7. Review the interactive charts in the ORB Visual Analysis section.
8. Change the year or month filters to compare historical ORB patterns across
   different periods.


## 14. Limitations

The NQ Futures ORB Price Action System has several limitations:

- The analysis and machine learning model are based on historical market data.
- Historical market behavior does not guarantee future trading results.
- The predictive model uses five selected market features and does not account
  for every factor that may influence NQ futures prices.
- Market behavior and relationships between features may change over time.
- The machine learning model has limited predictive accuracy on unseen test
  data and should be used only as a supplemental decision-support tool.
- The application does not use live market data.
- The application does not connect to a brokerage account or execute trades.


## 15. Disclaimer

The NQ Futures ORB Price Action System is intended for educational, analytical,
and decision-support purposes.

The application does not provide guaranteed trading results and should not be
considered financial advice. Futures trading involves substantial financial
risk. Users are responsible for evaluating market conditions and making their
own trading decisions.


## 16. Security, Monitoring, and Maintenance