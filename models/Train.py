import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('diabetes.csv')  # Replace with your dataset path
X = df[['Age', 'BMI', 'Glucose']]
y = df['Outcome']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Train another model (e.g., Logistic Regression, SVM, etc.)
from sklearn.linear_model import LogisticRegression
am_model = LogisticRegression(random_state=42)
am_model.fit(X_train, y_train)

# Save models
with open('models/random_forest.pkl', 'wb') as rf_file, \
     open('models/gradient_boosting.pkl', 'wb') as gb_file, \
     open('models/another_model.pkl', 'wb') as am_file:
    pickle.dump(rf_model, rf_file)
    pickle.dump(gb_model, gb_file)
    pickle.dump(am_model, am_file)

# Optionally, check accuracy for each model
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
am_pred = am_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))
print("Another Model Accuracy:", accuracy_score(y_test, am_pred))
