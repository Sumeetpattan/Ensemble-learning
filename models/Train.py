import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# Load dataset
df = pd.read_csv('diabetes.csv')  # Replace with the actual dataset path
X = df[['Age', 'BMI', 'Glucose']]
y = df['Outcome']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Train Logistic Regression Model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Save trained models
with open('models/random_forest.pkl', 'wb') as rf_file, \
     open('models/gradient_boosting.pkl', 'wb') as gb_file, \
     open('models/logistic_regression.pkl', 'wb') as lr_file:
    pickle.dump(rf_model, rf_file)
    pickle.dump(gb_model, gb_file)
    pickle.dump(lr_model, lr_file)

# Evaluate models on test data
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Print model accuracies
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
