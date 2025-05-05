import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load preprocessed data
X = pd.read_csv('datasets/preprocessed_X.csv')
y = pd.read_csv('datasets/preprocessed_y.csv')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train.values.ravel())  # Use .ravel() to convert y_train to 1D array

# Save model to models folder
joblib.dump(model, 'models/mental_health_model.pkl')
print("Mental health model trained and saved successfully!")
