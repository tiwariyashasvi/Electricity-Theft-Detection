# Electricity Theft Detection using Machine Learning

## Why I Made This

I was looking for a project idea that actually solves a real problem. Electricity theft is a huge issue in India and globally — power companies lose crores every year because of meter tampering and illegal connections. I thought if we have smart meter data, why not use AI to catch this automatically? That's how this project started.

## What This Project Does

It takes historical electricity consumption data of consumers and predicts whether a consumer is stealing electricity or not. Instead of manually checking thousands of bills, the AI model does it automatically.

## Dataset I Used

I used the SGCC dataset (State Grid Corporation of China) — it has daily electricity consumption records of 42,372 consumers over almost 3 years. I downloaded it from Kaggle.

Some quick facts about the data:
- 38,757 consumers are normal
- 3,615 consumers are marked as theft
- Each consumer has around 1,035 days of readings
- About 25% of the data had missing values

## How I Built It

**Step 1 — Data Cleaning**
First I removed all consumers who had more than 50% missing data because their records were too incomplete to be useful. This brought the dataset down from 42,372 to around 31,192 consumers.

**Step 2 — Understanding the Data**
I plotted the consumption graphs of normal vs theft consumers. You can clearly see the difference — normal consumers have consistent patterns while theft consumers show sudden drops, flat lines, or weird spikes.

**Step 3 — Feature Engineering**
Instead of feeding 1,035 raw daily readings into the model, I created 9 meaningful features that summarize each consumer's behavior:

| Feature | What it tells us |
|---|---|
| avg_consumption | Average daily usage |
| max_consumption | Highest usage day |
| min_consumption | Lowest usage day |
| std_consumption | How consistent the usage is |
| median_consumption | Middle value of daily usage |
| zero_days | How many days had zero usage |
| missing_days | How many days had no reading |
| range_consumption | Gap between highest and lowest day |
| peak_to_avg_ratio | Whether usage pattern looks tampered |

**Step 4 — Handling Class Imbalance**
The dataset has 91% normal and only 9% theft cases. If I trained directly on this, the model would just predict everyone as normal. I used SMOTE to create synthetic theft examples and balance the training data.

**Step 5 — Model Training**
I used Random Forest Classifier — it works by combining 100 decision trees and taking a majority vote. I split the data 80/20 for training and testing, making sure the theft ratio was maintained in both splits using stratification.

**Step 6 — Evaluation**
I didn't just use accuracy because with imbalanced data, accuracy is misleading. A model that predicts everyone as normal would still get 91% accuracy but catch zero thieves. So I focused on Recall, F1-Score and ROC-AUC.

| Metric | Score |
|---|---|
| Accuracy | 84.28% |
| ROC-AUC | 0.73 |
| Theft Recall | 38% |

**Step 7 — Web App**
I built an interactive web app using Streamlit where you can upload a CSV file of consumer data and instantly see which consumers are flagged as suspected theft, their risk percentage, consumption graphs, and key stats.

## How to Run

```bash
pip install -r requirements.txt
python eda.py
py -3.11 -m streamlit run app.py
```

## Tech Stack

- Python 3.11
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn
- Streamlit
- Matplotlib, Seaborn
- Random Forest Classifier
- SMOTE

## What I Learned

This project taught me that real world data is messy — missing values, imbalanced classes, and noisy readings are all things you have to deal with before even touching a model. Feature engineering matters more than the model itself sometimes. And evaluation metrics have to match the problem — accuracy alone can be completely misleading.
