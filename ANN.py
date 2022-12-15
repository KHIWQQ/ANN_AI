import pandas as pd
# import tensorflow
# from tensorflow import keras
import keras
# import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from numpy import loadtxt
# from keras.models import Sequential
# from keras.layers import Dense

# load the dataset
dataset = loadtxt('ABC.csv', delimiter=',')

# split into input (X) and output (y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
print(X)
print(Y)

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, Y, epochs=150, batch_size=100)

# evaluate the keras model
_, accuracy = model.evaluate(X, Y)

# accuracy = model.evaluate(X. Yverbose=0) tin case we do not need to  print out
print('Accuracy: %.2f' % (accuracy*100))

# make class predictions with the model
predictions = model.predict(X)

# round predictions
rounded = [round(x[0]) for x in predictions]