# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_csv('Fraud.csv')

# Data Cleaning
# Handle Missing Values
# Fill missing values with the median for numerical columns
numerical_columns = df.select_dtypes(include='number').columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# Visualize data to detect outliers
sns.boxplot(data=df)

# Feature Engineering
# Create new features: transaction_difference, origin_balance_diff, dest_balance_diff
df['transaction_difference'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['origin_balance_diff'] = df['amount'] - df['transaction_difference']
df['dest_balance_diff'] = df['newbalanceDest'] - df['oldbalanceDest']

# Encode categorical variables
# Convert categorical variables to numerical using label encoding
df['type'] = df['type'].astype('category').cat.codes
df['nameOrig'] = df['nameOrig'].astype('category').cat.codes
df['nameDest'] = df['nameDest'].astype('category').cat.codes

# Feature and Target Separation
X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df['isFraud']

# Train-Test Split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Feature Scaling
# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature Selection
# Use Recursive Feature Elimination (RFE) with XGBClassifier to select top features
model = XGBClassifier()
selector = RFE(model, n_features_to_select=10)
selector = selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Model Training
# Train the XGBoost classifier using selected features
model.fit(X_train_selected, y_train)

# Model Evaluation
# Predict on test set and calculate evaluation metrics
y_pred = model.predict(X_test_selected)
y_pred_proba = model.predict_proba(X_test_selected)

# Performance Metrics
# Print classification report, AUC-ROC score, confusion matrix, and accuracy
print(classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba[:, 1]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Feature Importance
# Calculate feature importances and create DataFrame
feature_importances = pd.DataFrame({'feature': X.columns[selector.support_], 'importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
print("Feature Importances:\n", feature_importances)

# Plot Feature Importances
# Visualize feature importances using a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Feature Importances')
plt.show()