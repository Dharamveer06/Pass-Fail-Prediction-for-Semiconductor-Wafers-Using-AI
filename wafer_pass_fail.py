import streamlit as st
import pandas as pd

st.set_page_config(page_title="Wafer Pass/Fail Predictor")

st.title("Wafer Pass/Fail Prediction App")

# Load dataset
try:
    df = pd.read_csv("wafer.csv")
    st.write("### Sample of wafer dataset:")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("⚠️ wafer.csv file not found. Please check if it's uploaded in the repo.")

# Add more logic or model loading below

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns



# Step 1: Load dataset
df = pd.read_csv("wafer.csv")

# Step 2: Explore and preprocess
print("Dataset shape:", df.shape)
print(df.head())

# Drop wafer id if any
if 'Wafer' in df.columns:
    df.drop('Wafer', axis=1, inplace=True)

# Step 3: Split features and label
X = df.drop('Pass_Fail', axis=1)
y = df['Pass_Fail']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
