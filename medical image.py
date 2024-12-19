import cv2
from sklearn.svm import SVC 
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,auc,ConfusionMatrixDisplay
from skimage.feature import hog

def image_preprocessing(folder_path, num_img):
    X = []
    labels = ['COVID19','NORMAL','PNEUMONIA']
    for folder_name in labels:
        label_folder = os.path.join(folder_path, folder_name)
        for filename in os.listdir(label_folder)[:num_img]:
            file_path = os.path.join(label_folder, filename)
            img = cv2.imread(file_path)
            res = cv2.resize(img, (256, 256));
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) 
            X.append(gray)
    return np.array(X)

def label(folder_path,num_img):
    y=[]
    labels = ['COVID19','NORMAL','PNEUMONIA']
    for label in labels:
        folder_name = label
        path = os.path.join(folder_path,folder_name)
        if os.path.isdir(path):
            y.extend([label] * min(num_img,len(os.listdir(path))))
    return np.array(y)
def hog_features(images):
        hog_features=[]
        for image in images:
             hog_feature = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), block_norm='L2-Hys',visualize = False,feature_vector=True)
             hog_features.append(hog_feature)

        return np.array(hog_features)

X_train_path = ""
X_test_path = ""
X_train = image_preprocessing(X_train_path,3418)
y_train = label(X_train_path,3418)
X_test = image_preprocessing(X_test_path,855)
y_test = label(X_test_path,855)

X_train_hog = hog_features(X_train)
X_test_hog = hog_features(X_test)
cv2.imshow('hog_image',X_train_hog)

svm_classifier = SVC(kernel='poly',C=1.0)
svm_classifier.fit(X_train_hog, y_train)

y_pred = svm_classifier.predict(X_test_hog)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test,y_pred)
classes = ['COVID19','NORMAL','PNEUMONIA']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot()
plt.title("Confusion Matrix")

plt.show()
print("Accuracy:", accuracy*100,"%")
