# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv('diabetes.csv')

# Display basic information about the dataset
print("\nDataset Information:")
print(f"Number of samples: {len(df)}")
print(f"Number of features: {len(df.columns)}")
print("\nFirst few rows of the dataset:")
print(df.head())

# Separate features (X) and target variable (y)
# The last column is the target variable (1 for diabetes, 0 for no diabetes)
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # Last column (target variable)

# Split the data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
print("\nTraining the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("\nModel Evaluation:")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance visualization
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importance in Diabetes Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance plot has been saved as 'feature_importance.png'")

# Example of making a prediction for a new patient
print("\nExample of making a prediction for a new patient:")
# Create a sample patient data (using mean values from the dataset)
sample_patient = X.mean().values.reshape(1, -1)
sample_patient_scaled = scaler.transform(sample_patient)
prediction = model.predict(sample_patient_scaled)
probability = model.predict_proba(sample_patient_scaled)

print(f"Prediction: {'Diabetes' if prediction[0] == 1 else 'No Diabetes'}")
print(f"Probability of diabetes: {probability[0][1]:.2f}") 