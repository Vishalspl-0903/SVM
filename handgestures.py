import cv2
from sklearn.svm import SVC 
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
from skimage.feature import hog
import seaborn as sns


def preprocess(folder_path,num_images):
    X = []
    for label in range(20): 
        label_folder = os.path.join(folder_path, str(label))
        for filename in os.listdir(label_folder)[:num_images]:
            if filename.endswith('.jpg'):
                filepath = os.path.join(label_folder, filename)
                image = cv2.imread(filepath)  
                resize = cv2.resize(image, (64, 64))
                gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY) 
                cv2.imshow('Image',gray)
                X.append(gray)
                
    return np.array(X)


def labelling(folder_path,num_images):
    y=[]
    for label in range(20):
        folder_name = str(label)
        path = os.path.join(folder_path,folder_name)
        if os.path.isdir(path):
            y.extend([label] * min(num_images,len(os.listdir(path))))
    return np.array(y)


def hog_features(images):
        hog_features=[]
        for image in images:
             hog_feature = hog(image, orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2), block_norm='L2-Hys',visualize = False,feature_vector=True)
             hog_features.append(hog_feature)

        return np.array(hog_features)


X_train_path = ""
X_test_path = ""
X_train = preprocess(X_train_path,800)
y_train = labelling(X_train_path,800)
X_test = preprocess(X_test_path,200)
y_test = labelling(X_test_path,200)


X_train_hog = hog_features(X_train)
X_test_hog = hog_features(X_test)
svm_classifier = SVC(kernel='sigmoid',C=1.5,gamma=0.05) 
svm_classifier.fit(X_train_hog, y_train)

y_pred = svm_classifier.predict(X_test_hog)
accuracy = accuracy_score(y_test, y_pred) 
cm = confusion_matrix(y_test,y_pred) 
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Reds', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.show() 
print("Accuracy:", accuracy*100,"%")

image = input("Enter the image path: ")
img = cv2.imread(image)  
resize = cv2.resize(img, (64, 64))
gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY) 
hog_feature = hog(gray, orientations=9, pixels_per_cell=(4, 4),
                   cells_per_block=(2, 2), block_norm='L2-Hys',
                   visualize=False, feature_vector=True)
hog_feature = hog_feature.reshape(1, -1)  
pred_num = svm_classifier.predict(hog_feature)
print("The image you inputted is denoting the number:", pred_num)











