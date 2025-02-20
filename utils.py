import pickle
import pandas as pd
import numpy as np

def load_models():
    """
    Load trained machine learning models from pickle files.
    
    Returns:
    - List of loaded models [Random Forest, Gradient Boosting, Logistic Regression]
    """
    try:
        # Load models
        with open('models/random_forest.pkl', 'rb') as rf_file:
            rf_model = pickle.load(rf_file)
        with open('models/gradient_boosting.pkl', 'rb') as gb_file:
            gb_model = pickle.load(gb_file)
        with open('models/logistic_regression.pkl', 'rb') as lr_file:  # Updated to Logistic Regression
            lr_model = pickle.load(lr_file)
        
        print("Models loaded successfully.")
        return [rf_model, gb_model, lr_model]
    
    except FileNotFoundError as e:
        print(f"Error: Model file not found - {e}")
        return []
    except Exception as e:
        print(f"Error loading models: {e}")
        return []

def preprocess_input(data):
    """
    Preprocess the input data from the user for prediction.

    Args:
    - data (dict): Input data from the user.

    Returns:
    - pd.DataFrame: Preprocessed features in a DataFrame.
    """
    try:
        # Extract features from the input data
        age = float(data['age'])
        bmi = float(data['bmi'])
        glucose = float(data['glucose'])

        # Create a DataFrame to match the expected input format for the model
        features = pd.DataFrame([[age, bmi, glucose]], columns=['Age', 'BMI', 'Glucose'])

        return features
    
    except KeyError as e:
        print(f"Error: Missing key in input data - {e}")
        return None
    except ValueError as e:
        print(f"Error: Invalid value in input data - {e}")
        return None
