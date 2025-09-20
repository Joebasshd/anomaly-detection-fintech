import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import logging
import pickle  # Importing pickle to save the model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tune_isolation_forest(X_scaled, param_grid):
    """
    Tune the hyperparameters of IsolationForest to find the best combination.
    
    Args:
        X_scaled (pd.DataFrame): The scaled feature matrix.
        param_grid (dict): Dictionary containing the grid of hyperparameters to search.
    
    Returns:
        list: Results containing the tuning outcomes.
    """
    results = []

    # Hyperparameter grid search
    for n in param_grid['n_estimators']:
        for max_samp in param_grid['max_samples']:
            for cont in param_grid['contamination']:
                try:
                    clf = IsolationForest(
                        n_estimators=n,
                        max_samples=max_samp,
                        contamination=cont,
                        random_state=42,
                        n_jobs=-1
                    )
                    clf.fit(X_scaled)  # Fit the model to the scaled data

                    scores = clf.decision_function(X_scaled)
                    anomalies = clf.predict(X_scaled)  # -1 for anomalies

                    top_anom_idx = np.argsort(scores)[:50]
                    top_anomalies = X_scaled.iloc[top_anom_idx]  # Get top anomalies

                    results.append({
                        'n_estimators': n,
                        'max_samples': max_samp,
                        'contamination': cont,
                        'mean_score': np.mean(scores),
                        'num_flagged': np.sum(anomalies == -1),
                        'top_anomalies_preview': top_anomalies.head(3)
                    })

                    logger.info(f"Finished tuning: n_estimators={n}, max_samples={max_samp}, contamination={cont}")

                except Exception as e:
                    logger.error(f"Error during Isolation Forest tuning: {e}")
    
    return results


def fit_isolation_forest(X_scaled_df, n_estimators=100, max_samples=128, contamination=0.01):
    """
    Fit IsolationForest with given hyperparameters and predict anomalies.
    
    Args:
        X_scaled_df (pd.DataFrame): The scaled feature matrix.
        n_estimators (int): Number of base estimators.
        max_samples (int): Number of samples used for each base estimator.
        contamination (float): Proportion of outliers in the data.
    """
    try:
        isf_model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, random_state=42)
        isf_model.fit(X_scaled_df)

        # Predict anomaly scores
        anomaly_scores = isf_model.decision_function(X_scaled_df)

        # Predict anomaly labels (-1 for anomalies and 1 for inliers)
        anomaly_labels = isf_model.predict(X_scaled_df)

        X_scaled_df['anomaly_score'] = anomaly_scores
        X_scaled_df['anomaly_label'] = anomaly_labels

        # Save the trained model using pickle
        with open("isolation_forest_model.pkl", "wb") as model_file:
            pickle.dump(isf_model, model_file)

        logger.info(f"Isolation Forest model fitted with n_estimators={n_estimators}, max_samples={max_samples}, contamination={contamination}")
    
    except Exception as e:
        logger.error(f"Error during Isolation Forest fitting: {e}")


def get_anomalies_from_isolation_forest(df):
    """
    Get transactions flagged as anomalies by Isolation Forest.
    
    Args:
        df (pd.DataFrame): The DataFrame containing anomaly labels from Isolation Forest.
    
    Returns:
        pd.DataFrame: A DataFrame with anomalies flagged by the Isolation Forest model.
    """
    try:
        anomalies = df[df['anomaly_label'] == -1]
        logger.info(f"Found {len(anomalies)} anomalies detected by Isolation Forest.")
        return anomalies

    except Exception as e:
        logger.error(f"Error during anomaly extraction: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


def main(X_scaled, X_scaled_df):
    """
    Main function to run hyperparameter tuning for Isolation Forest, fit the model, and find anomalies.
    
    Args:
        X_scaled (pd.DataFrame): The scaled feature matrix for hyperparameter tuning.
        X_scaled_df (pd.DataFrame): The scaled feature matrix for Isolation Forest.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_samples': [128, 256],
        'contamination': [0.01, 0.02]
    }

    tuning_results = tune_isolation_forest(X_scaled, param_grid)
    
    # Fit IsolationForest using chosen hyperparameters
    fit_isolation_forest(X_scaled_df, n_estimators=100, max_samples=128, contamination=0.01)

    anomalies = get_anomalies_from_isolation_forest(X_scaled_df)

    return anomalies
