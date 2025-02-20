from flask import Flask, request, render_template, jsonify
import sqlite3
import pandas as pd
import numpy as np
from utils import preprocess_input, load_models

app = Flask(__name__)

# Initialize SQLite database
DATABASE = 'database.db'

def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        age INTEGER,
                        gender TEXT,
                        bmi REAL,
                        glucose REAL,
                        prediction TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Load pre-trained models
models = []
try:
    models = load_models()
    if models:
        print("Models loaded successfully.")
    else:
        print("Warning: No models were loaded.")
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Parse input data
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        required_fields = ['name', 'age', 'gender', 'bmi', 'glucose']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Log received data
        print(f"Received data: {data}")

        # Preprocess input features
        features = preprocess_input(data)
        if features is None:
            return jsonify({"error": "Invalid input format"}), 400
        
        print(f"Processed features: {features}")

        # Ensure models are loaded
        if not models:
            return jsonify({"error": "Models are not available. Please try again later."}), 500

        # Generate predictions from models
        predictions = [model.predict(features)[0] for model in models]
        print(f"Model predictions: {predictions}")

        # Convert predictions to native Python int for JSON serialization
        predictions = [int(pred) for pred in predictions]
        print(f"Converted predictions: {predictions}")

        # Perform majority voting for final prediction
        final_prediction = max(set(predictions), key=predictions.count)
        print(f"Final prediction: {final_prediction}")

        # Map numeric prediction to "Low risk" or "High risk"
        risk_map = {0: "Low risk", 1: "High risk"}
        final_prediction_text = risk_map.get(final_prediction, "Unknown")

        # Save results to the database
        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO users (name, age, gender, bmi, glucose, prediction)
                              VALUES (?, ?, ?, ?, ?, ?)''',
                           (data['name'], data['age'], data['gender'], data['bmi'], data['glucose'], final_prediction_text))
            conn.commit()
        except sqlite3.Error as db_error:
            print(f"Database error: {db_error}")
            return jsonify({"error": "Failed to save prediction to database."}), 500
        finally:
            conn.close()

        # Return the prediction result
        return jsonify({"prediction": final_prediction_text})

    except KeyError as e:
        print(f"Missing key: {e}")
        return jsonify({"error": f"Missing key: {e}"}), 400
    except ValueError as e:
        print(f"Value error: {e}")
        return jsonify({"error": "Invalid data format"}), 400
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
