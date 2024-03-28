import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

# dir = 'C:\\Users\\ADMIN\\Desktop\\SVM for Skin type\\skin-dataset'
# categories = ['dry', 'normal','oily']

# data = []
# for category in categories:
#     cnt = 0
#     path = os.path.join(dir, category)
#     for img in os.listdir(path):
#         cnt += 1
#         img_array = cv2.imread(os.path.join(path, img))
#         #cv2.imshow('image', img_array)
#         try:
#             img_array = cv2.resize(img_array, (50, 50))
#             image = np.array(img_array).flatten()
#             data.append([image, categories.index(category)])
#         except Exception as e:
#             pass
    
#     print(f'{category} : {cnt} images')

# print(len(data))
# pick_in = open('data.pickle', 'wb')
# pickle.dump(data, pick_in)
# pick_in.close()

pick_in = open('data.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.5)

# model = SVC(C=1, kernel='poly', gamma='auto')
# model.fit(xtrain, ytrain)

# pick = open('model.sav', 'wb')
# pickle.dump(model, pick)
# pick.close()

pick = open('model.sav', 'rb')
model = pickle.load(pick)
pick.close()

# # Load the uploaded image
# uploaded_image = cv2.imread('Thumbnail-6.jpg')  

# # Preprocess the image (resize, flatten)
# resized_image = cv2.resize(uploaded_image, (50, 50))  
# flattened_image = np.array(resized_image).flatten()  
# Make prediction using the trained SVM model
#prediction = model.predict([flattened_image])

categories = ['dry', 'normal','oily']

# prediction = model.predict(xtest)
# # accuracy = model.score(xtest, ytest)
# # f1 = f1_score(ytest, prediction, average='weighted')
# accuracy = accuracy_score(ytest, prediction)
# f1 = f1_score(ytest, prediction, average='weighted')

# print(f'Accuracy: {accuracy}')
# #print(f'Prediction: {categories[prediction[0]]}')
# print(f'F1 Score: {f1}')


cv_scores = cross_val_score(model, features, labels, cv=5)  

# Calculate mean accuracy and F1 score
mean_accuracy = np.mean(cv_scores)
mean_f1 = np.mean(f1_score(labels, model.predict(features), average=None))

print(f'Accuracy: {mean_accuracy}')
print(f'F1 Score: {mean_f1}')

#skin = flattened_image
#skin = np.array(xtest[0]).reshape(50, 50,3)
# plt.imshow(skin)
# plt.show()