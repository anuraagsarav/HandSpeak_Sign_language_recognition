import os
import json
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Path where dataset is stored
DATA_PATH = 'dataset'

# Define signs (labels)
SIGNS = [folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))]

# Load dataset
X, y = [], []
for sign in SIGNS:
    sign_folder = os.path.join(DATA_PATH, sign)
    for file in os.listdir(sign_folder):
        if file.endswith('.json'):
            file_path = os.path.join(sign_folder, file)
            with open(file_path, 'r') as f:
                landmarks = json.load(f)
                landmark_flattened = [coord for point in landmarks for coord in point.values()]
                X.append(landmark_flattened)
                y.append(sign)

# Convert to numpy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model_path = 'models/sign_language_model.pkl'
if not os.path.exists('models'):
    os.makedirs('models')
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)
print(f"Model saved to {model_path}")

