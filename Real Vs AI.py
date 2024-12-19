import cv2
from sklearn.svm import SVC 
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
from skimage.feature import hog
import seaborn as sns

def preprocess_images(folder_path, num_images):
    X = []
    y = []
    for filename in os.listdir(folder_path)[:num_images]:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)
            resize = cv2.resize(image, (64, 64))
            X.append(resize) 
            if filename[0] == '0':
                y.append(1)  
            else:
                y.append(0) 
    return np.array(X), np.array(y)

train_fake_path = ""
train_real_path = ""
test_fake_path = ""
test_real_path = ""


X_train_fake, y_train_fake = preprocess_images(train_fake_path, 50000)
X_train_real, y_train_real = preprocess_images(train_real_path, 50000)
X_test_fake, y_test_fake = preprocess_images(test_fake_path, 10000)
X_test_real, y_test_real = preprocess_images(test_real_path, 10000)
 
X_train = np.concatenate((X_train_fake, X_train_real))
y_train = np.concatenate((y_train_fake, y_train_real))
X_test = np.concatenate((X_test_fake, X_test_real))
y_test = np.concatenate((y_test_fake, y_test_real))

def extract_hog_features(images):
    hog_features = []
    
    for image in images:
        if len(image.shape) > 2 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
 
        hog_feature = hog(gray, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), block_norm='L2-Hys',visualize = False,feature_vector=True)
        hog_features.append(hog_feature)

    return np.array(hog_features)

X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

svm_classifier = SVC(kernel='rbf',C=1.5, gamma=0.1)
svm_classifier.fit(X_train_hog, y_train)

y_pred = svm_classifier.predict(X_test_hog)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test,y_pred)
TN, FP, FN, TP = cm.ravel()
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Greens', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
print("True Positives (TP):", TP)
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)    
print("False Negatives (FN):", FN)
plt.show()
print("Accuracy:", accuracy*100,"%")

