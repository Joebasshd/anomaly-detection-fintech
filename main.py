from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import logging
from typing import Optional

import pandas as pd
# import shap

# Attempt to import shap.  If it is missing, install it on the fly.
try:
    import shap  # type: ignore[import]
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap  # type: ignore[import]

# Local imports
import parser as log_parser
import features
import model as anomaly_model
from explain import explain_anomaly_naturally

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_device_and_location_features_fixed(df: pd.DataFrame) -> None:
    """Add device and location change features with a robust implementation."""
    logger.info("Adding device and location features...")
    df['is_new_location'] = df.groupby('user_id')['location'].transform(
        lambda x: (x != x.shift()).astype(int)
    )

    mismatch_series = (
        df.groupby('user_id')
        .apply(
            lambda x: (
                (x['location'] != x['location'].shift())
                & (x['device'] != x['device'].shift())
            ).astype(int)
        )
        .reset_index(level=0, drop=True)
    )
    df['geo_device_mismatch'] = mismatch_series

    df['amount_zscore'] = df.groupby('user_id')['txn_amount'].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    df['amount_to_avg_ratio'] = df['txn_amount'] / df['avg_user_transaction_amount']


def _custom_preprocess_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess and scale features with a patched device/location handler."""
    logger.info("Preprocessing and scaling features...")
    df_working = df.copy()
    df_working['orig_idx'] = df_working.index

    user_features = features.engineer_user_features(df_working)
    df_working = df_working.merge(user_features, on='user_id', how='left')

    features.add_time_based_features(df_working)
    features.add_transaction_frequency_features(df_working)

    add_device_and_location_features_fixed(df_working)

    orig_indices = df_working['orig_idx'].tolist()

    df_working = df_working.drop(
        columns=[col for col in ['timestamp', 'user_id', 'orig_idx'] if col in df_working.columns]
    )

    categorical_cols = df_working.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df_working, columns=categorical_cols, drop_first=False)

    df_encoded = features.drop_low_variance_features(df_encoded, threshold=0.01)

    df_encoded = features.drop_highly_correlated_features(
        df_encoded, correlation_threshold=0.9
    )

    df_scaled = features.scale_features(df_encoded)

    return df_scaled, orig_indices


def process_file(
    input_path: str,
    top_n: int = 3,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Run the anomaly detection and explainability pipeline on a CSV file."""
    logger.info(f"Processing file: {input_path}")
    if not os.path.isfile(input_path):
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df_input = pd.read_csv(input_path)

    # Determine if we should parse raw logs
    if "raw_log" in df_input.columns:
        logger.info("Parsing raw logs...")
        df_parsed = log_parser.load_and_parse_logs(input_path)
    else:
        df_parsed = df_input.copy()

    preprocess_result = _custom_preprocess_and_scale(df_parsed)
    if isinstance(preprocess_result, tuple):
        X_scaled_df, orig_indices = preprocess_result
    else:
        X_scaled_df = preprocess_result
        orig_indices = list(range(len(X_scaled_df)))

    X_scaled_df.index = orig_indices
    X_features_df = X_scaled_df.copy()

    logger.info("Fitting Isolation Forest model...")
    anomaly_model.fit_isolation_forest(X_scaled_df)

    anomalies_df = anomaly_model.get_anomalies_from_isolation_forest(X_scaled_df)
    if anomalies_df.empty:
        logger.info("No anomalies detected by Isolation Forest.")
        if output_path:
            pd.DataFrame().to_csv(output_path, index=False)
        return pd.DataFrame()

    # Load model and explainability
    logger.info("Loading the trained Isolation Forest model and generating explanations...")
    with open("isolation_forest_model.pkl", "rb") as model_file:
        isolation_model = pickle.load(model_file)

    explainer = shap.TreeExplainer(isolation_model)
    shap_values = explainer.shap_values(X_features_df)

    explanations: list[str] = []
    for row_idx in anomalies_df.index:
        shap_row = shap_values[row_idx]
        instance = X_features_df.loc[row_idx]
        explanation = explain_anomaly_naturally(
            instance=instance,
            shap_vals=shap_row,
            feature_names=X_features_df.columns,
            top_n=top_n,
            X_original=None,
        )
        explanations.append(explanation)

    anomalies_df = anomalies_df.copy()
    anomalies_df.loc[:, "explanation"] = explanations

    merged = pd.concat(
        [df_parsed.loc[anomalies_df.index].reset_index(drop=True),
         anomalies_df[["anomaly_score", "anomaly_label", "explanation"]].reset_index(drop=True)],
        axis=1,
    )

    out_path = output_path or "anomaly_explanations.csv"
    merged.to_csv(out_path, index=False)

    logger.info(f"Processed anomalies saved to {out_path}")
    return merged


def main() -> None:
    """Entry point for command line execution."""
    parser = argparse.ArgumentParser(
        description="Run anomaly detection with explainability on a transaction CSV file."
    )
    parser.add_argument(
        "input",
        help="Path to the input CSV file (raw logs or structured data)"
    )
    parser.add_argument(
        "--output",
        dest="output",
        default=None,
        help="Optional path to write the output CSV with explanations."
    )
    parser.add_argument(
        "--top_n",
        dest="top_n",
        type=int,
        default=3,
        help="Number of top features to include in each explanation (default: 3)"
    )

    args = parser.parse_args()
    logger.info("Starting process...")
    process_file(
        input_path=args.input,
        top_n=args.top_n,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
