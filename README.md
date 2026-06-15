# Electricity Theft Detection using Machine Learning

I built this project to learn how AI can be applied in the energy 
sector. The idea was to detect electricity theft by analyzing smart 
meter consumption data and flagging unusual patterns automatically.

## Dataset
I used the SGCC dataset which has consumption records of 42,372 customers.
Download `data.csv` from [Kaggle](https://www.kaggle.com/datasets/sreen28g10/electricity-theft-detection) 
and place it in the same folder before running.

## What I did
- Cleaned the data and handled missing values
- Visualized normal vs theft consumption patterns
- Trained an Isolation Forest model to detect anomalies
- Evaluated results using confusion matrix and classification report

## Results
- Accuracy: 87.08%
- Precision: 23.50%
- Recall: 22.82%
- F1-Score: 23.16%

## How to run
pip install pandas numpy matplotlib seaborn scikit-learn
python electricity_theft_detection.py
