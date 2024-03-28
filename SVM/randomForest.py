import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Define a function to extract features from images (e.g., color histograms)
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image '{image_path}'")
        return None
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # Flatten histogram to 1D array
    return hist

# Path to your dataset folder
dataset_folder = 'C:\\Users\\ADMIN\\Desktop\\SVM for Skin type\\skin-dataset'

# List to store features and labels
features = []
labels = []

# Iterate over subfolders (oily, dry, and normal)
for skin_type in os.listdir(dataset_folder):
    skin_type_folder = os.path.join(dataset_folder, skin_type)
    if os.path.isdir(skin_type_folder):
        # Iterate over each image file in the subfolder
        for filename in os.listdir(skin_type_folder):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                # Extract features from the image
                image_path = os.path.join(skin_type_folder, filename)
                image_features = extract_features(image_path)
                if image_features is not None:
                    # Add features and corresponding label to lists
                    features.append(image_features)
                    labels.append(skin_type)

# Convert lists to numpy arrays
X = np.array(features)
y = np.array(labels)

# Check if X or y is empty
if len(X) == 0 or len(y) == 0:
    print("Error: No data found")
    exit()

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predicting labels for the test set
y_pred = rf_classifier.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculating F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)