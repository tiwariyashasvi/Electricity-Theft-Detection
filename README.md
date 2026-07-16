# Electricity Theft Detection — GridGuard

This is a machine learning project that predicts electricity theft from consumer consumption data. You upload a CSV of daily consumption readings, and the app cleans the data, calculates a set of statistics for each consumer, and runs them through a trained Random Forest model to flag suspicious consumers.

It's built with Streamlit, so everything runs in the browser — upload, cleaning, prediction, and visualization all happen in one page.

## What it does

- Takes a CSV with consumer IDs and daily consumption values
- Removes consumers with more than 50% missing data (their usage pattern can't be trusted otherwise)
- Calculates 9 features per consumer (average consumption, standard deviation, zero-days, etc.)
- Predicts Normal vs Theft using a trained Random Forest model, along with a confidence percentage
- Shows an overview of the results (theft rate, distribution chart)
- Lets you pick any individual consumer and see their daily consumption graph
- Explains why a specific consumer was flagged, by comparing their numbers to the rest of the dataset instead of using fixed thresholds

## Tech used

- Streamlit for the app/UI
- pandas and numpy for data handling
- scikit-learn for the Random Forest model
- matplotlib for the graphs
- joblib to save/load the trained model

## Files in this repo

- `app.py` — the Streamlit app itself
- `feature_engineering.py` — cleaning logic + feature calculation
- `train_model.py` — script used to train the model
- `electricity_theft_model.joblib` — the trained model
- `requirements.txt` — Python packages needed
- `confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`, `sample_consumption_patterns.png` — plots from training/evaluation

Note: `data.csv` isn't included here because it was too large for GitHub. You just upload your own CSV through the app directly.

## How to run it

Clone the repo:

```
git clone https://github.com/tiwariyashasvi/Electricity-Theft-Detection.git
cd Electricity-Theft-Detection
```

Install the requirements:

```
pip install -r requirements.txt
```

Run the app:

```
streamlit run app.py
```

It'll open in your browser at `localhost:8501`.

## Expected CSV format

| Column | What it is |
|---|---|
| CONS_NO | consumer ID |
| FLAG | theft label used for training (1 = theft, 0 = normal) |
| daily columns (e.g. 2014/1/1, 2014/1/2...) | daily kWh consumption |

## The 9 features used by the model

- **avg_consumption** — average daily usage
- **max_consumption** — highest single day
- **min_consumption** — lowest single day
- **std_consumption** — how much usage fluctuates
- **median_consumption** — middle value, less affected by spikes than the average
- **zero_days** — how many days had exactly 0 kWh recorded
- **missing_days** — how many days had no reading at all
- **range_consumption** — max minus min
- **peak_to_avg_ratio** — max divided by average, shows how spiky the usage is
