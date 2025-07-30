# 💳 Fraud Detection using Isolation Forest and XGBoost
This project focuses on detecting fraudulent transactions using Machine Learning techniques. Fraud detection is a critical problem in finance, and accurate prediction models can help prevent large financial losses. In this project, we use both Isolation Forest (an unsupervised anomaly detection algorithm) and XGBoost (a supervised gradient boosting algorithm) to identify fraudulent transactions.

## 🧠 Objective
Build and compare machine learning models to detect fraudulent transactions in a highly imbalanced dataset, focusing on precision and recall metrics to reduce false positives and negatives.

## 📂 Dataset
The dataset used is the popular Fraud Detection dataset from Accredian, which contains:

6362620 transactions

812 fraudulent transactions (≈ 0.17%)

Features are anonymized using PCA (except Time and Amount)

## 🔄 Workflow Overview
This project follows the typical Data Science pipeline:

1. Data Collection
Data is sourced from Accredian Fraud Dataset and loaded into a pandas DataFrame.

2. Data Preprocessing
Missing values check: No missing values found.

Data types: All features are numeric.

Scaling: Amount and Time features were scaled using StandardScaler.

Class imbalance: Addressed through anomaly detection and evaluation strategies rather than oversampling/undersampling.

3. Exploratory Data Analysis (EDA)
Visualized class imbalance using bar plots.

Distribution plots and correlation heatmaps used to understand relationships.

Observed that fraud transactions often have smaller amounts and different feature patterns.

4. Modeling
Two different approaches were used:

🔹 A. Isolation Forest (Unsupervised)
Isolation Forest works by isolating anomalies in the dataset.

Since fraudulent data is rare, it’s treated as an outlier.

Model trained on all data without labels.

Contamination rate set close to 0.0017 (the fraud rate).

Predictions compared with actual labels.

🔹 B. XGBoost (Supervised)
Trained using labeled data.

Handled class imbalance using the scale_pos_weight parameter.

Performed train-test split (80:20).

Tuned hyperparameters using GridSearchCV for optimal performance.

Feature importance plotted to understand key drivers of fraud detection.

5. Model Evaluation
Used Precision, Recall, F1-Score, and ROC-AUC to evaluate models.

Due to class imbalance, Accuracy is misleading.

Focused on Recall (sensitivity) to minimize missed frauds.

Plotted confusion matrices and ROC curves.

Metric	Isolation Forest	XGBoost
Precision	~0.30	~0.91
Recall	~0.65	~0.87
ROC-AUC	~0.89	~0.99

🔍 Key Insights
Isolation Forest is useful for anomaly detection without labels but less precise.

XGBoost performs significantly better in identifying fraudulent transactions.

Balancing precision and recall is crucial—especially in financial contexts where false positives can also be costly.

📦 Technologies Used
Python

Pandas, NumPy – data handling

Matplotlib, Seaborn – data visualization

Scikit-learn – preprocessing and Isolation Forest

XGBoost – supervised learning model

Imbalanced-learn – handled class imbalance

🚀 How to Run
Clone this repo:

<p>
git clone https://github.com/yourusername/fraud-detection-model.git
cd fraud-detection-model
</p>

Run the notebook:
Open fraud_detection.ipynb in Jupyter Notebook or run scripts from terminal.

📝 Future Work
Implement SMOTE for oversampling.

Try deep learning models (e.g., Autoencoders).

Deploy model using Flask or FastAPI.

Integrate with real-time transaction APIs.

📌 Conclusion
Fraud detection is a high-impact problem with serious real-world consequences. In this project, we used both unsupervised (Isolation Forest) and supervised (XGBoost) methods to tackle the problem. While unsupervised methods are useful when labels are unavailable, supervised models like XGBoost provide much better performance when trained on labeled data.


