import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and preprocessor
MODEL_PATH = os.path.join('artifacts', 'model.pkl')
PREPROCESSOR_PATH = os.path.join('artifacts', 'preprocessor.pkl')

with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

with open(PREPROCESSOR_PATH, 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # Render a simple HTML form for input

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form or API
        data = request.form if request.form else request.json

        # Convert input data to a DataFrame
        input_data = pd.DataFrame([data])

        # Preprocess the input data
        processed_data = preprocessor.transform(input_data)

        # Make predictions
        prediction = model.predict(processed_data)

        # Decode the prediction (if necessary)
        result = 'Income > 50K' if prediction[0] == 1 else 'Income <= 50K'

        # Return the result
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)