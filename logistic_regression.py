import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

# Logistic regression. Data splits


#1 Побудувати модель логістичної регресії із регуляризацією для будь якого набору даних

dataset = pd.read_csv('logistic regression dataset-Social_Network_Ads.csv')
dataset['Gender'] = dataset['Gender'].map({'Female': 1, 'Male': 0})
print(dataset.head())

x = np.asarray(dataset.drop(['User ID', 'Purchased'], axis=1))
y = dataset['Purchased']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.fit_transform(x_test)

model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
model.fit(x_train_scaler, y_train)

y_pred = model.predict(x_test_scaler)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
y_prob = model.predict_proba(x_test_scaler)
loss = log_loss(y_test, y_prob)

print('Accuracy:', accuracy)
print('Confusion matrix:', conf_matrix)
print('Log loss:', loss)










