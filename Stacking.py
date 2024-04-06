import os
import cv2
import numpy as np
import efficientnet.keras as efn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize EfficientNetB0 model
effnet_model = efn.EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

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

# Path to the dataset folder
dataset_folder = '/content/drive/MyDrive/skin-dataset/skin-dataset'

# List to store features and labels
features = []
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
                features_effnet = extract_features_effnet(image_path)
                if features_effnet is not None:
                    # Add features and corresponding label to lists
                    features.append(features_effnet)
                    labels.append(label)

# Convert lists to numpy arrays
X_effnet = np.array(features)
y = np.array(labels)

# Splitting the dataset into train and test sets
X_effnet_train, X_effnet_test, y_train, y_test = train_test_split(X_effnet, y, test_size=0.2, random_state=42)

# Initialize base classifiers (Random Forest, SVC, EfficientNetB0)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='linear', C=1.0)

# Train base classifiers
random_forest.fit(X_effnet_train, y_train)
svm.fit(X_effnet_train, y_train)

# Predictions from base classifiers on the test set
y_pred_rf = random_forest.predict(X_effnet_test)
y_pred_svm = svm.predict(X_effnet_test)

# Stack predictions from base classifiers to form input for meta learner
X_meta_train = np.column_stack((y_pred_rf, y_pred_svm))

# Initialize meta learner (e.g., logistic regression)
meta_learner = LogisticRegression()

# Train meta learner on the stacked predictions
meta_learner.fit(X_meta_train, y_test)

# Predictions from base classifiers on the training set
y_train_pred_rf = random_forest.predict(X_effnet_train)
y_train_pred_svm = svm.predict(X_effnet_train)

# Stack predictions from base classifiers on the training set to form input for meta learner
X_meta_test = np.column_stack((y_train_pred_rf, y_train_pred_svm))

# Predictions from meta learner on the stacked predictions
y_pred_meta = meta_learner.predict(X_meta_test)

# Calculate accuracy of the stacked ensemble model
accuracy_stacked_ensemble = accuracy_score(y_train, y_pred_meta)
print("Stacked Ensemble Model Accuracy:", accuracy_stacked_ensemble)
