#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import necessary libraries
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)

# Dictionary to store deployed models
deployed_models = {}

# Route to train and deploy a model
@app.route('/train_and_deploy', methods=['POST'])
def train_and_deploy():
    try:
        # Get data and parameters from the request
        data = request.json
        model_id = data.get('model_id')
        test_size = data.get('test_size', 0.2)

        # Load dataset (Assuming data is provided in the request)
        # ... (Code to load and preprocess data)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train a model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save the model
        model_filename = f"{model_id}_model.joblib"
        joblib.dump(model, model_filename)

        # Update deployed_models dictionary
        deployed_models[model_id] = {
            'model': model,
            'accuracy': accuracy,
            'version': len(deployed_models) + 1
        }

        return jsonify({'message': f'Model trained and deployed with accuracy: {accuracy}'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to make predictions using a deployed model
@app.route('/predict/<model_id>', methods=['POST'])
def predict(model_id):
    try:
        # Check if the model_id is valid
        if model_id not in deployed_models:
            return jsonify({'error': 'Model not found'}), 404

        # Load the deployed model
        model = deployed_models[model_id]['model']

        # Get input data from the request
        input_data = request.json.get('input_data')

        # Make predictions
        predictions = model.predict(input_data)

        return jsonify({'predictions': predictions.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)


# In[ ]:




