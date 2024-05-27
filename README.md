# Fraud_Detection_Model
This model id developed for predicting fraudulent transactions for a financial company and use insights from the model to develop an actionable plan.

### Fraud Detection Model:
    - The fraud detection model is based on an XGBoost classifier, which is a type of gradient boosting algorithm.
    - XGBoost is known for its high performance and scalability, making it suitable for handling large datasets and capturing 
      complex relationships between features.
    - The model is trained on features derived from transaction data, including engineered features such as transaction 
      difference, origin balance difference, and destination balance difference.

### Data cleaning including missing values, outliers and multi-collinearity. 
    - Missing Values: Missing values in numerical columns were filled with the median value of each column.
    - Outliers: The code for outlier removal using z-score was commented out. Alternatively, a boxplot visualization was used to 
      detect outliers visually. Outlier removal was not explicitly performed in the provided code.
    - Multi-collinearity: Multi-collinearity was not addressed explicitly in the provided code. It refers to the situation where 
      predictor variables in a regression model are highly correlated with each other. Techniques like variance inflation 
      factor (VIF) or correlation matrix analysis can be used to identify and address multi-collinearity.

### Variable Selection:
    - Recursive Feature Elimination (RFE) with XGBoost was used for variable selection.
    - RFE recursively removes features and selects the optimal subset of features that contribute the most to the model's 
      performance.
    - In the provided code, RFE was configured to select the top 10 features based on their importance.

### Model Performance:
    - The performance of the model was evaluated using various metrics including classification report, AUC-ROC score,
      confusion matrix and accuracy.
    - These metrics provide insights into the model's ability to correctly classify fraudulent and non-fraudulent
      transactions, as well as its overall predictive performance.
      AUC-ROC: 0.9997241886163819
      Confusion Matrix:
      [[1906233      89]
      [    369    2095]]
      Accuracy: 0.999760056915757

### Key Factors Predicting Fraudulent Customer:
    - These features include transaction characteristics, account balances, transaction differences, and other relevant 
      information derived from the dataset.

