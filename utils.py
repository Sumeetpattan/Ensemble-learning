import pickle

def load_models():
    try:
        # Load models from pickle files
        with open('models/random_forest.pkl', 'rb') as rf_file:
            rf_model = pickle.load(rf_file)
        with open('models/gradient_boosting.pkl', 'rb') as gb_file:
            gb_model = pickle.load(gb_file)
        with open('models/another_model.pkl', 'rb') as am_file:  # Add your new model here
            am_model = pickle.load(am_file)
        return [rf_model, gb_model, am_model]
    except Exception as e:
        print(f"Error loading models: {e}")
        return []
    
import pandas as pd
import numpy as np

def preprocess_input(data):
    """
    Preprocess the input data from the user for prediction.

    Args:
    - data (dict): Input data from the user.

    Returns:
    - pd.DataFrame: Preprocessed features in a DataFrame.
    """
    # Extract features from the input data
    age = float(data['age'])
    bmi = float(data['bmi'])
    glucose = float(data['glucose'])

    # Create a DataFrame to match the expected input format for the model
    features = pd.DataFrame([[age, bmi, glucose]], columns=['Age', 'BMI', 'Glucose'])

    # Example: You can add more preprocessing steps here like scaling, encoding, etc.
    # For now, let's return the features as-is.

    return features
