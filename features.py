import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer

# Define helper functions for feature engineering

def calculate_entropy(series):
    """Calculates the entropy of a pandas Series."""
    value_counts = series.value_counts(normalize=True)
    entropy = -np.sum(value_counts * np.log(value_counts))
    return entropy

def engineer_user_features(df):
    """Generate aggregated features per user."""
    grouped_user_data = df.groupby('user_id')

    # Feature engineering for user-based aggregations
    user_features = pd.DataFrame({
        'avg_user_transaction_amount': grouped_user_data['txn_amount'].mean(),
        'std_user_transaction_amount': grouped_user_data['txn_amount'].std(),
        'user_max_transaction_amount': grouped_user_data['txn_amount'].max(),
        'transaction_type_entropy': grouped_user_data['txn_type'].apply(calculate_entropy),
        'device_count': grouped_user_data['device'].nunique(),
        'location_count': grouped_user_data['location'].nunique()
    })

    # Drop device_count and location_count after merging
    user_features.drop(columns=['device_count', 'location_count'], inplace=True)

    return user_features

def add_time_based_features(df):
    """Generate time-based features (hour, day, night, etc.)."""
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['hour_of_day_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_of_day_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = np.where(df['day_of_week'].isin([5, 6]), 1, 0)
    df['is_night'] = np.where((df['hour_of_day'] >= 21) | (df['hour_of_day'] <= 4), 1, 0)
    df.drop('hour_of_day', axis=1, inplace=True)

def add_transaction_frequency_features(df):
    """Add transaction frequency and time delta features."""
    df.sort_values(by=['user_id', 'timestamp'], inplace=True)
    df['time_since_last_txn'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
    df.fillna({'time_since_last_txn': 0}, inplace=True)

    # Transaction count in the last 24 hours
    txn_count_24hr_series = df.groupby('user_id').rolling('24h', on='timestamp')['timestamp'].count().reset_index(drop=True)
    df['txn_count_24hr'] = txn_count_24hr_series

    # Location change frequency (relative to total transactions per user)
    location_changes = df.groupby('user_id')['location'].apply(lambda x: (x != x.shift()).sum() - 1)
    total_transactions = df.groupby('user_id').size()
    location_change_freq = location_changes / total_transactions
    df['location_change_freq'] = df['user_id'].map(location_change_freq)

def add_device_and_location_features(df):
    """Generate device and location change features."""
    most_common_location = df.groupby('user_id')['location'].apply(lambda x: x.mode()[0])
    df['is_new_location'] = df.groupby('user_id')['location'].transform(lambda x: (x != x.shift()).astype(int))
    df['geo_device_mismatch'] = df.groupby('user_id')[['location', 'device']].apply(
        lambda x: (x['location'] != x['location'].shift()) & (x['device'] != x['device'].shift())
    ).astype(int).reset_index(level=0, drop=True)

    df['amount_zscore'] = df.groupby('user_id')['txn_amount'].transform(lambda x: (x - x.mean()) / x.std())
    df['amount_to_avg_ratio'] = df['txn_amount'] / df['avg_user_transaction_amount']

def preprocess_features(df):
    """Main function to preprocess the features."""
    user_features = engineer_user_features(df)
    df = df.merge(user_features, on='user_id', how='left')
    
    add_time_based_features(df)
    add_transaction_frequency_features(df)
    add_device_and_location_features(df)

    cols_to_drop = [col for col in ['timestamp', 'user_id'] if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    categorical_cols = df.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    return df_encoded

def drop_low_variance_features(df, threshold=0.01):
    """Automatically drop features with variance below the given threshold."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    low_variance_cols = numeric_cols[df[numeric_cols].var() < threshold]
    df.drop(columns=low_variance_cols, inplace=True)
    return df

def drop_highly_correlated_features(df, correlation_threshold=0.9):
    """Drop features with high correlation based on a given threshold."""
    correlation_matrix = df.corr()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
    
    df.drop(columns=to_drop, inplace=True)
    return df

def scale_features(df):
    """Scale the features using StandardScaler, MinMaxScaler, and RobustScaler."""

    standard_features = ['amount_zscore', 'hour_of_day_sin', 'hour_of_day_cos']
    minmax_features = ['txn_amount', 'day_of_week', 'is_weekend', 'is_night', 'is_new_location', 'geo_device_mismatch']
    robust_features = ['avg_user_transaction_amount', 'std_user_transaction_amount', 'user_max_transaction_amount',
                       'transaction_type_entropy', 'time_since_last_txn', 'txn_count_24hr', 'location_change_freq',
                       'amount_to_avg_ratio']

    used_features = set(standard_features + minmax_features + robust_features)
    categorical_features = [col for col in df.columns if col not in used_features]

    preprocessor = ColumnTransformer(transformers=[ 
        ('std', StandardScaler(), [f for f in standard_features if f in df.columns]),
        ('mm', MinMaxScaler(), [f for f in minmax_features if f in df.columns]),
        ('rob', RobustScaler(), [f for f in robust_features if f in df.columns]),
        ('cat', 'passthrough', categorical_features)
    ], remainder='drop')

    X_scaled = preprocessor.fit_transform(df)

    scaled_feature_names = (
        [f for f in standard_features if f in df.columns] +
        [f for f in minmax_features if f in df.columns] +
        [f for f in robust_features if f in df.columns] +
        categorical_features
    )
    return pd.DataFrame(X_scaled, columns=scaled_feature_names)

def main(df):
    """Main function to preprocess and scale features."""
    df_encoded = preprocess_features(df)
    
    df_encoded = drop_low_variance_features(df_encoded, threshold=0.01)
    df_encoded = drop_highly_correlated_features(df_encoded, correlation_threshold=0.9)

    df_scaled = scale_features(df_encoded)
    
    return df_scaled