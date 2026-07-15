import pandas as pd
import numpy as np


NON_CONSUMPTION_COLUMNS = ["CONS_NO", "FLAG"]

FEATURE_DESCRIPTIONS = {
    "avg_consumption": "Average daily electricity consumption (kWh) over the full recorded period.",
    "max_consumption": "The single highest daily consumption value recorded for this consumer.",
    "min_consumption": "The single lowest daily consumption value recorded for this consumer.",
    "std_consumption": "Standard deviation of daily consumption, capturing how volatile or stable usage is.",
    "median_consumption": "The middle value of daily consumption, less sensitive to extreme spikes than the average.",
    "zero_days": "Number of days where recorded consumption was exactly zero, which can indicate meter tampering or non-usage.",
    "missing_days": "Number of days where consumption data was missing (NaN) for this consumer.",
    "range_consumption": "The difference between maximum and minimum consumption (max - min), showing the spread of usage.",
    "peak_to_avg_ratio": "Ratio of maximum consumption to average consumption. Unusually high or low ratios can indicate abnormal usage patterns.",
}

FEATURE_DISPLAY_NAMES = {
    "avg_consumption": "Average Consumption",
    "max_consumption": "Maximum Consumption",
    "min_consumption": "Minimum Consumption",
    "std_consumption": "Standard Deviation",
    "median_consumption": "Median Consumption",
    "zero_days": "Zero Consumption Days",
    "missing_days": "Missing Days",
    "range_consumption": "Consumption Range",
    "peak_to_avg_ratio": "Peak-to-Average Ratio",
}

FEATURE_ORDER = [
    "avg_consumption",
    "max_consumption",
    "min_consumption",
    "std_consumption",
    "median_consumption",
    "zero_days",
    "missing_days",
    "range_consumption",
    "peak_to_avg_ratio",
]


def get_consumption_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_CONSUMPTION_COLUMNS]


def compute_missing_percentage(df: pd.DataFrame) -> pd.Series:
    consumption_cols = get_consumption_columns(df)
    return df[consumption_cols].isnull().mean(axis=1) * 100


def remove_high_missing_consumers(df: pd.DataFrame, threshold: float = 50.0):
    missing_percentage = compute_missing_percentage(df)
    keep_mask = missing_percentage <= threshold
    df_clean = df[keep_mask].copy()
    removed_df = df[~keep_mask].copy()
    return df_clean, removed_df, missing_percentage


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    consumption_cols = get_consumption_columns(df)
    consumption = df[consumption_cols]

    features_df = pd.concat([
        consumption.mean(axis=1).rename("avg_consumption"),
        consumption.max(axis=1).rename("max_consumption"),
        consumption.min(axis=1).rename("min_consumption"),
        consumption.std(axis=1).rename("std_consumption"),
        consumption.median(axis=1).rename("median_consumption"),
        (consumption == 0).sum(axis=1).rename("zero_days"),
        consumption.isnull().sum(axis=1).rename("missing_days"),
        (consumption.max(axis=1) - consumption.min(axis=1)).rename("range_consumption"),
    ], axis=1)

    features_df["peak_to_avg_ratio"] = (
        features_df["max_consumption"] / (features_df["avg_consumption"] + 1e-5)
    )

    features_df = features_df[FEATURE_ORDER]
    features_df.index = df.index
    return features_df


def full_preprocessing_pipeline(df: pd.DataFrame, threshold: float = 50.0):
    df_clean, removed_df, missing_percentage = remove_high_missing_consumers(df, threshold)
    features_df = create_features(df_clean)
    return df_clean, removed_df, missing_percentage, features_df
