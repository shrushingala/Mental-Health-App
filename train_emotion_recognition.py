import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load FER2013 dataset from datasets folder
df = pd.read_csv('datasets/FER2013.csv')
X = np.array([np.fromstring(x, sep=' ') for x in df['pixels']])
y = df['emotion']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model (e.g., Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model to models folder
joblib.dump(model, 'models/emotion_recognition_model.pkl')
print("Emotion recognition model trained and saved successfully!")
