import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('datasets/RHMCD-20.csv')

# Initialize LabelEncoder for each column
label_encoders = {}

# Encode all columns
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Save the encoder for each column

    # Save the LabelEncoder for the target variable ('Mental_Health_History')
    if column == 'Mental_Health_History':
        joblib.dump(le, 'models/mental_health_history_label_encoder.pkl')

# Separate features and target variable
X = df.drop('Mental_Health_History', axis=1)
y = df['Mental_Health_History']

# Save preprocessed data
X.to_csv('datasets/preprocessed_X.csv', index=False)
y.to_csv('datasets/preprocessed_y.csv', index=False)
mental_health_model = joblib.load('models/mental_health_model.pkl')
# Save all label encoders for future use (if needed)
joblib.dump(label_encoders, 'models/label_encoders.pkl')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Predict on the test set
y_pred = mental_health_model.predict(X_test)
print("Preprocessing complete. Preprocessed data saved as preprocessed_X.csv and preprocessed_y.csv.")
print("Label encoders saved as label_encoders.pkl.")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Mental Health Model: {accuracy}")
