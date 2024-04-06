import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from efficientnet.keras import EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Initialize EfficientNetB0 model
effnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features using EfficientNetB0
def extract_features_effnet(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image '{image_path}'")
        return None
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # EfficientNet expects RGB images
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    features = effnet_model.predict(image)
    return features.flatten()

# Function to extract features from images using Random Forest and SVM
def extract_features_rf_svm(image_path, target_size=(100, 100)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image '{image_path}'")
        return None
    image = cv2.resize(image, target_size)
    return image.flatten()

# Path to the dataset folder
dataset_folder = '/content/drive/MyDrive/skin-dataset/skin-dataset'

# List to store features and labels
features_rf_svm = []
features_effnet = []
labels = []

# Mapping for skin types to labels
label_mapping = {"dry": 0, "normal": 1, "oily": 2}

# Iterate over subfolders (dry, normal, and oily)
for skin_type in os.listdir(dataset_folder):
    skin_type_folder = os.path.join(dataset_folder, skin_type)
    if os.path.isdir(skin_type_folder):
        label = label_mapping.get(skin_type)
        # Iterate over each image file in the subfolder
        for filename in os.listdir(skin_type_folder):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                # Extract features from the image using EfficientNetB0
                image_path = os.path.join(skin_type_folder, filename)
                features_effnet.append(extract_features_effnet(image_path))
                # Extract features from the image using Random Forest and SVM
                features_rf_svm.append(extract_features_rf_svm(image_path))
                labels.append(label)

# Convert lists to numpy arrays
X_rf_svm = np.array(features_rf_svm)
X_effnet = np.array(features_effnet)
y = np.array(labels)

# Splitting the dataset into train and test sets for RF and SVM
X_rf_svm_train, X_rf_svm_test, y_train, y_test = train_test_split(X_rf_svm, y, test_size=0.2, random_state=42)

# Splitting the dataset into train and test sets for EfficientNetB0
X_effnet_train, X_effnet_test, _, _ = train_test_split(X_effnet, y, test_size=0.2, random_state=42)

# Initialize base classifiers
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel='linear', C=1.0)

# Train base classifiers
rf_model.fit(X_rf_svm_train, y_train)
svm_model.fit(X_rf_svm_train, y_train)

# Predictions for base classifiers
y_pred_rf = rf_model.predict(X_rf_svm_test)
y_pred_svm = svm_model.predict(X_rf_svm_test)

# Initialize XGBoost classifier
xgb_classifier = xgb.XGBClassifier()

# Train XGBoost classifier on predictions of base classifiers
xgb_classifier.fit(np.column_stack((y_pred_rf, y_pred_svm)), y_test)

# Predictions using XGBoost classifier
y_pred_xgb = xgb_classifier.predict(np.column_stack((y_pred_rf, y_pred_svm)))

# Calculate accuracy of the ensemble model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Ensemble Model Accuracy:", accuracy_xgb)
