# **Anomaly Detection System for Transaction Monitoring**

This project implements an **anomaly detection system** designed to identify unusual or potentially fraudulent transactions in a dataset of user activity. The system uses machine learning models, including **Isolation Forest** and **DBSCAN**, to detect anomalies in transaction data. The project also includes a simple CLI deployment.


## **Project Overview**

The anomaly detection system is designed to analyze user transaction data and flag transactions that exhibit **unusual behavior**. This is achieved through a series of steps:

1. **Data Parsing**: The raw transaction data is parsed and cleaned, extracting key features like timestamps, transaction amounts, locations, and devices.
2. **Feature Engineering**: Various features are engineered based on transaction behavior, such as the number of transactions per user, the average transaction amount, and time-based features (e.g., is it a weekend?).
3. **Modeling**: The **Isolation Forest** and **DBSCAN** models are trained on the engineered features to detect anomalies. The Isolation Forest model is the primary model used for predicting anomalies in deployment, with DBSCAN serving for additional insight into the data.
4. **Explainability**: The **SHAP** method is used to generate human-readable explanations for the flagged anomalies, providing transparency into the model’s decisions.
5. **Deployment**: The model is deployed using **FastAPI** (for model inference) and **Streamlit** (for the user interface), allowing for easy interaction with the system for both uploading transaction data and receiving predictions.


### **Parsing Logic Explanation**

The parsing logic is designed to process and clean raw transaction logs, converting them into structured data suitable for feature engineering and model training. Here’s a detailed breakdown of the key steps involved in the parsing process:

1. **Raw Log Parsing**:

   * The system is capable of handling **two different log formats**:

     * **Strict format**: The first format is a `::`-delimited log format (e.g., `timestamp::user::txn_type::amount::location::device`), which is parsed by splitting the log line into its components.
     * **Fallback format**: The second format is a more flexible one, where the log could have different delimiters or structures. In this case, we attempt to extract relevant fields (timestamp, user, transaction type, amount, location, device) using **regular expressions**.

2. **Feature Extraction**:

   * **Timestamp**: The timestamp is extracted and converted into a standard ISO format using a list of predefined date formats. This step ensures that the timestamps are consistent across the dataset.
   * **User ID**: The user identifier is extracted by matching patterns like `usr:userXXXX` or `userXXXX` where `XXXX` is the user number.
   * **Transaction Type**: The system identifies the transaction type (e.g., withdrawal, deposit, purchase, etc.) by searching for specific keywords using regular expressions.
   * **Transaction Amount**: The amount of the transaction is extracted by matching numeric patterns with floating point values.
   * **Location**: Locations are extracted using a regular expression that identifies and captures location names. A list of stopwords (like "at", "in", etc.) is also handled to improve accuracy.
   * **Device**: The device used for the transaction is extracted from the part of the string that comes after the location. A special function `extract_device_after_location` helps to parse the device information. It also handles cases where the device information might be preceded by certain keywords like `device=`, `dev:`, etc.

3. **Handling Missing Data**:

   * If any of the fields (location or device) cannot be parsed, they are explicitly set to the string `"None"`, rather than the Python `None` object. This helps avoid any confusion during downstream processes and keeps the data consistent, especially when dealing with missing or malformed entries.

4. **General Fallback Parsing**:

   * If the strict parsing format doesn't work or the log line is malformed, the function falls back to using the general pattern-matching approach. This allows the system to handle **various irregular log formats** while still extracting key fields.

5. **Final Output**:

   * After parsing, the function returns a dictionary containing the structured fields (`timestamp`, `user_id`, `txn_type`, `txn_amount`, `location`, `device`), which are then added to a DataFrame for further processing.

6. **Cleaning and Imputing Missing Data**:

   * The parser also filters out empty logs and rows titled **malformed logs** based on certain predefined markers (e.g., `MALFORMED_LOG`, `""`, `"`, etc.). Any logs that are incomplete or not matching the expected format are skipped, ensuring that the dataset only contains valid entries for further analysis.

The parser’s design makes it robust to handle a variety of transaction log formats, correctly extract the necessary features, and clean up any missing or malformed entries before they are used for machine learning and analysis.


## **Modeling Approach**

The system uses two machine learning models for anomaly detection:

### **1. Isolation Forest**

* **Isolation Forest** is used to detect anomalies based on the assumption that anomalies are fewer and different from the rest of the data. This model is trained on the features extracted from the transaction data and assigns each transaction a score indicating its likelihood of being anomalous.
* **Hyperparameter Tuning**: The model’s hyperparameters (number of estimators, max samples, contamination) were tuned using a grid search approach to optimize for precision and ensure accurate anomaly detection.

### **2. DBSCAN**

* **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) is used for clustering the data. Transactions that are outliers, according to the density criterion, are flagged as anomalies.
* **Limitations**: DBSCAN was less effective for this task and flagged a kot of transactions. As a result, DBSCAN was used for **insights** and **outlier exploration** rather than as the primary model for deployment.

### **Intersection of Predictions**

* The **Intersection of Anomalies** is derived by selecting transactions flagged as anomalous by both **Isolation Forest** and **DBSCAN**. This set of transactions is considered the most suspicious and likely requires further investigation.

### **Model Explainability**

* **SHAP (SHapley Additive exPlanations)** was used to explain the model’s decisions. SHAP values provide an interpretable explanation of the model’s predictions by attributing each feature’s contribution to the anomaly score.
* **Natural Language Explanations**: The SHAP values are converted into human-readable explanations, such as "This transaction was flagged as anomalous because: the transaction occurred in Glasgow, which decreased the anomaly score."


## **Findings**

### **User Behavior**

* A small subset of users (user1001, user1015, user1088) account for a significant portion of anomalies. These users exhibit **repetitive and unusual transaction behavior**, suggesting that they may be involved in **fraudulent activities** or system errors.
* **Debit transactions** were flagged most frequently, especially **withdrawals and transfers**, indicating that **financial transactions** are the most likely to trigger anomalies.

### **Location and Device**

* Transactions from certain locations (e.g., Glasgow, Cardiff) and devices (e.g., Huawei P30, Samsung Galaxy S10) are more likely to be flagged as anomalous, especially if **location or device data is missing**.
* The model detects **geo-device mismatches**, where both location and device change simultaneously, as a strong indicator of anomalies.

### **Time-Based Features**

* Anomalies are more likely to occur on **weekends** (with a higher occurrence of flagged transactions) and shortly after the **previous transaction** (e.g., within 24 hours).
* **Time gaps** and **transaction frequency** are important features, with anomalies occurring more frequently shortly after previous transactions.

### **Feature Importance**

* **Location** and **device one-hot encoded features** are the most influential in detecting anomalies, followed by **transaction type**.
* **Geo-device mismatch** (a simultaneous change in location and device) is a strong indicator of anomalies.
* **Weekend transactions** are more likely to be flagged, with anomalies primarily occurring during the weekend.


## **Business Impact**

1. **Fraud Detection**:

   * **Debit transactions**, particularly **withdrawals and transfers**, should be closely monitored, as they are more likely to be flagged as anomalies.
   * **Location and device changes** should be flagged, especially for **high-value transactions**.
   * **Weekend transactions** should receive additional scrutiny as they are more likely to be anomalies.

2. **Improving Data Quality**:

   * **Missing location or device data** leads to a higher number of flagged anomalies. Ensuring complete and accurate data collection for these fields will improve the model’s performance.

3. **Regular Model Retraining**:

   * The model’s reliance on location and device features means that it may be prone to **false positives** when deployed in new regions or with new devices. **Retraining the model regularly** and improving the feature engineering pipeline will ensure it remains adaptable to new user behavior.

4. **Operational Efficiency**:

   * The intersection of anomalies detected by both **Isolation Forest** and **DBSCAN** identifies the most suspicious transactions. This approach allows for more **efficient investigation** by focusing on the transactions that are most likely to represent fraudulent activity.



## **How to Run the System**

   ```bash
 python main.py synthetic_dirty_transaction_logs.csv --output output.csv
   ```


