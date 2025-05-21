import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, request, jsonify, render_template
import os

# Load dataset
import pandas as pd

# Correct file path format
file_path = r"C:\Users\Jaisw\OneDrive\Desktop\Warranty_Claim_Dataset.csv"

# Read the dataset
df = pd.read_csv(file_path)

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)


print(df.head())


def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

outlier_cols = ['Claim_Value', 'Service_Centre', 'Product_Age', 'Call_details']
for col in outlier_cols:
    df = remove_outliers(df, col)

# Encoding categorical variables
df = pd.get_dummies(df, drop_first=True)

# Splitting data
X = df.drop(columns=['Fraud'])
y = df['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Flask App
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        claim_amount = float(request.form['claim_amount'])
        product_age = int(request.form['product_age'])
        customer_history = int(request.form['customer_history'])
        purchase_date = request.form['purchase_date']
        claim_date = request.form['claim_date']
        claim_reason = request.form['claim_reason']
        repair_cost = float(request.form['repair_cost'])
        warranty_validity = int(request.form['warranty_validity'])

        # Dummy prediction logic 
        prediction = "Fraudulent" if claim_amount > 5000 else "Genuine"

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

