# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Load the dataset
            df = pd.read_csv('Mall_Customers.csv')  # Update with your dataset path

            # Data preprocessing (update this based on your dataset)
            X = df.drop('Annual Income (k$)', axis=1)
            y = df['Annual Income (k$)']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a basic model (update this based on your requirements)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Make predictions on test data
            y_pred = model.predict(X_test)

            # Evaluate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            return render_template('result.html', accuracy=accuracy)

        except Exception as e:
            return str(e)

if __name__ == '__main__':
    app.run(debug=True)
