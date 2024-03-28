import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import random
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# #Define the paths to your train and test data directories
# train_dir = 'C:\\Users\\ADMIN\\Desktop\\SVM for Skin type\\Oily-Dry-Skin-Types\\train'
# test_dir = 'C:\\Users\\ADMIN\\Desktop\\SVM for Skin type\\Oily-Dry-Skin-Types\\test'

# #List of class labels
# categories = ['dry', 'oily', 'normal']

# # Initialize an empty list to store training and testing data
# train_data = []
# test_data = []

# # Load training data
# for category_id, category in enumerate(categories):
#     cnt = 0
#     category_dir = os.path.join(train_dir, category)
#     for img_file in os.listdir(category_dir):
#         cnt += 1
#         img_path = os.path.join(category_dir, img_file)
#         img_array = cv2.imread(img_path)
#         # cv2.imshow('image', img_array)
#         # break
#         try:
#             img_array = cv2.resize(img_array, (50, 50))
#             img_flattened = img_array.flatten()
#             train_data.append([img_flattened, category_id])
#         except Exception as e:
#             pass

#     print(f'{category} : {cnt} images')

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # Load testing data

# for category_id, category in enumerate(categories):
#     cnt = 0
#     category_dir = os.path.join(test_dir, category)
#     for img_file in os.listdir(category_dir):
#         cnt += 1
#         img_path = os.path.join(category_dir, img_file)
#         img_array = cv2.imread(img_path)
#         try:
#             img_array = cv2.resize(img_array, (50, 50))
#             img_flattened = img_array.flatten()
#             test_data.append([img_flattened, category_id])
#         except Exception as e:
#             pass

#     print(f'{category} : {cnt} images')

# pick_in = open('dataSkinTrain.pickle', 'wb')
# pickle.dump(train_data, pick_in)
# pick_in.close()

# pick_in_test = open('dataSkinTest.pickle', 'wb')
# pickle.dump(test_data, pick_in_test)
# pick_in_test.close()

pick_in = open('dataSkinTrain.pickle', 'rb')
train_data = pickle.load(pick_in)
pick_in.close()

pick_in_test = open('dataSkinTest.pickle', 'rb')
test_data = pickle.load(pick_in_test)
pick_in_test.close()

# Shuffle the data
random.shuffle(train_data)
random.shuffle(test_data)

# Separate features and labels for training and testing data
train_features = [data[0] for data in train_data]
train_labels = [data[1] for data in train_data]
test_features = [data[0] for data in test_data]
test_labels = [data[1] for data in test_data]

# # Train the model
# model = SVC(C=1, kernel='poly', gamma='auto')
# model.fit(train_features, train_labels)

# pick = open('modelSkin.sav', 'wb')
# pickle.dump(model, pick)
# pick.close()

# Load the pre-trained model
pick = open('modelSkin.sav', 'rb')
model = pickle.load(pick)
pick.close()

# Make predictions on the testing data
prediction = model.predict(test_features)

# Calculate accuracy
accuracy = model.score(test_features, test_labels)

f1 = f1_score(test_labels, prediction, average=None)

categories = ['dry', 'oily', 'normal']

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Prediction: {categories[prediction[0]]}')

# Sample image from the testing data
sample_image = test_features[0].reshape(50, 50,3)
plt.imshow(sample_image)
plt.show()
