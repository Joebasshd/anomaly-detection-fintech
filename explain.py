from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explain_anomaly_naturally(
    instance: pd.Series,
    shap_vals: Sequence[float],
    feature_names: Iterable[str],
    top_n: int = 3,
    X_original: pd.DataFrame | None = None,
) -> str:
    """Generate a natural language explanation for a flagged anomalous transaction.
    Returns a human‑readable explanation summarising why the transaction was
        flagged as anomalous.
    """
    logger.info("Starting explanation process...")

    # Ensure inputs are aligned
    if len(shap_vals) != len(feature_names):
        logger.error(f"Length mismatch: {len(shap_vals)} SHAP values vs {len(feature_names)} feature names.")
        raise ValueError(
            "Length of shap_vals does not match number of feature_names."
        )

    feature_contributions: list[tuple[str, float, float]] = []
    instance_values = instance.loc[list(feature_names)].values

    logger.info("Analyzing feature contributions...")

    for feature, shap_val, value in zip(feature_names, shap_vals, instance_values):
        feature_contributions.append((feature, float(shap_val), value))

    sorted_features = sorted(
        feature_contributions,
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    reasons: list[str] = []
    for feature, shap_val, value in sorted_features:
        if len(reasons) >= max(1, top_n):
            break

        if (
            (feature.startswith("location_") or feature.startswith("txn_type_") or feature.startswith("device_") or feature.startswith("most_common_location_"))
            and (value == 0 or pd.isna(value))
        ):
            continue

        direction = "increased" if shap_val > 0 else "decreased"
        clean_feature_name = feature.replace("_", " ").capitalize()

        if feature.startswith("location_"):
            original_location = feature[len("location_"):].replace("_", " ").capitalize()
            reason = (
                f"the transaction occurred in {original_location}, "
                f"which {direction} the anomaly score"
            )
        elif feature.startswith("most_common_location_"):
            original_location = feature[len("most_common_location_"):].replace("_", " ").capitalize()
            reason = (
                f"the user typically transacts in {original_location}, "
                f"which {direction} the score"
            )
        elif feature.startswith("txn_type_"):
            txn_type = feature[len("txn_type_"):].replace("_", " ").capitalize()
            reason = f"the transaction was a {txn_type}, which {direction} the score"
        elif feature.startswith("device_"):
            device = feature[len("device_"):].replace("_", " ").capitalize()
            reason = (
                f"the device used was {device}, "
                f"which {direction} the anomaly score"
            )
        else:
            if isinstance(value, (int, float, np.integer, np.floating)):
                formatted_value = f"{float(value):.2f}"
            else:
                formatted_value = str(value)
            reason = f"{clean_feature_name} = {formatted_value} {direction} the anomaly score"

        reasons.append(reason)

    if not reasons and sorted_features:
        logger.warning("No reasons were found; using top feature as fallback explanation.")
        feature, shap_val, value = sorted_features[0]
        direction = "increased" if shap_val > 0 else "decreased"
        clean_feature_name = feature.replace("_", " ").capitalize()
        if isinstance(value, (int, float, np.integer, np.floating)):
            formatted_value = f"{float(value):.2f}"
        else:
            formatted_value = str(value)
        reasons.append(f"{clean_feature_name} = {formatted_value} {direction} the anomaly score")

    explanation = (
        "This transaction was flagged as anomalous because: "
        + "; ".join(reasons)
        + "."
    )
    
    logger.info("Explanation generated successfully.")
    return explanation

__all__ = ["explain_anomaly_naturally"]
