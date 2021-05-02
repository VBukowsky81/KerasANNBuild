# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Sample_Dataset.csv')
X = dataset.iloc[:, 3:12].values
y = dataset.iloc[:, 12].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building ANN

from keras.models import Sequential
from keras.layers import Dense

# Initiliazing ANN model object
classifier = Sequential()

# Adding Layers
classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu'))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu'))

classifier.add(Dense(units=1, kernel_initializer='uniform', activation = 'sigmoid')) # or 'softmax' if we have MULTIPLE outcomes, here only 2

# Compiling ANN model
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting ANN to the Training Set(or Data)
classifier.fit(X_train, y_train, batch_size = 10, epochs=100)

# Part 3 - Predicting Results

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
