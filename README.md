# Heart-Disease-prediction-using-ML-models
Heart Disease Prediction using Machine Learning
Project Overview

This project predicts the presence of heart disease in patients using a dataset containing various clinical attributes. Multiple machine learning algorithms, including Decision Tree and K-Nearest Neighbors (KNN), are used to build predictive models and evaluate their performance.

Dataset

The dataset used is heart.csv.

It contains 13 features (predictors) and 1 target variable (target).

Features description:

Feature	Description
age	Age of the patient
sex	1: male, 0: female
cp	Chest pain type (0–3)
trestbps	Resting blood pressure
chol	Serum cholesterol in mg/dl
fbs	Fasting blood sugar > 120 mg/dl (1: true; 0: false)
restecg	Resting electrocardiographic results (0–2)
thalach	Maximum heart rate achieved
exang	Exercise induced angina (1: yes; 0: no)
oldpeak	ST depression induced by exercise relative to rest
slope	Slope of the peak exercise ST segment
ca	Number of major vessels (0–3) colored by fluoroscopy
thal	Defect type (0–3)
target	Heart disease presence (1: disease, 0: no disease)
Project Steps
1. Data Exploration & Visualization

Histograms for all features to understand distributions.

Barplots and pairplots to visualize relationships between categorical features and target.

Kernel Density Estimation (KDE) plots for numerical features.

Correlation heatmap to detect feature dependencies.

2. Data Preprocessing

Split dataset into features X and target y.

Split data into training set (70%) and test set (30%).

Standardized features for KNN using StandardScaler.

3. Machine Learning Models

Decision Tree Classifier

Trained on the training dataset.

Predictions made on the test dataset.

Accuracy calculated for both training and test sets.

Feature importance visualized to interpret model.

K-Nearest Neighbors (KNN)

KNN model trained on standardized features.

Accuracy evaluated for different values of k to find the optimal number of neighbors.

4. Model Evaluation

Accuracy scores of all models were compared using a bar plot.

Example prediction demonstrates the usage of trained models on a new patient input.

Results
Algorithm	Accuracy (%)
Decision Tree	(example: 85%)
KNN	(example: 82%)

Feature importance in Decision Tree highlighted cp (chest pain), thalach (max heart rate), and oldpeak (ST depression) as important predictors.

Example Usage
import numpy as np

# Example patient input
X_sample = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]])
prediction = dt.predict(X_sample)

if prediction[0] == 1:
    print("Yes - Heart Disease Detected 💔")
else:
    print("No - Healthy Heart ❤️")

Libraries Used

numpy

pandas

matplotlib

seaborn

plotly

scikit-learn

Future Improvements

Test additional models like Random Forest, Logistic Regression, SVM, LightGBM, XGBoost for higher accuracy.

Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

Deploy the model as a web app for real-time predictions.

Add feature engineering and dimensionality reduction for improved performance.

License

This project is open-source and free to use for educational purposes.
