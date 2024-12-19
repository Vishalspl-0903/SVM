import re
import nltk;
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve,auc

dataset = pd.read_csv(r'')
def text_preprocessing(text):
    text = re.sub(r'\W',' ',text) 
    text = text.lower() 
    text = re.sub(r'\s+',' ',text) 
    text = re.sub(r'\d',' ',text) 
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

dataset['preprocessed_reviews'] = dataset['review'].apply(text_preprocessing)
dataset['sentiment'] = dataset['sentiment'].apply(lambda sentiment: 1 if sentiment == 'positive' else 0)
X = dataset['preprocessed_reviews']
y = dataset['sentiment']
vector_converter = TfidfVectorizer(max_features=10000)
X = vector_converter.fit_transform(X).toarray()
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=45)
imdb_model = SVC(kernel='linear',probability=True)
imdb_model.fit(X_train, y_train)
y_pred = imdb_model.predict(X_test)
y_pred_prob = imdb_model.predict_proba(X_test)[:, 1] 
accuracy = accuracy_score(y_test, y_pred)  
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel() 
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob) 
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title('Confusion Matrix')

print("True Positives (TP):", TP)
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)
plt.show()

print("Accuracy:", accuracy*100,"%")
