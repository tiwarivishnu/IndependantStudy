# -*- coding: utf-8 -*-
"""
Created on Saturday July 27 17:13:06 2018
@author: Vishnu Tiwari - MSBAIS USF
"""
# References
import os, sys, math
import keras
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_context("notebook", font_scale=1.4)
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from livelossplot import PlotLossesKeras
'exec(%matplotlib inline)'

#Data Load

data = pd.read_csv("C:/VT/Independent Study/ran_peps_netMHCpan40_predicted_A0201_reduced_cleaned_balanced.csv",
                   sep='\t')

x = data.drop(columns=["label_num", "data_type", "label_chr"])
y = data.drop(columns=["peptide", "label_chr", "data_type"])

#Inspect Data
print(x)
print(y)

#Peptide encoding method

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def show_matrix(m):
          cm = sns.light_palette("seagreen", as_cmap=True)

def one_hot_encode(seq):
    o = list(set(codes) - set(seq))
    s = pd.DataFrame(list(seq))
    x = pd.DataFrame(np.zeros((len(seq),len(o)),dtype=int),columns=o)
    a = s[0].str.get_dummies(sep=',')
    a = a.join(x)
    a = a.sort_index(axis=1)
    e = a.values.flatten()
    return e

#Test the peptide encoding method
pep = 'LLTDAQRIV'
e = one_hot_encode(pep)
print(e)
print(len(e))
p = np.reshape(e, (20, 9, 1))
print(p)

# Encoding of all the peptides. create an object to hold loop results
print("length of x:" + str(len(x)))
x = x.values
print("x length after removing header: " + str(len(x)))
x_loop = []
for i in x:
    x_loop.append(one_hot_encode(str(i)[2:11]))

print("x_loop length: " + str(len(x_loop)))

#Reassign the values x is features and y is the output
x = x_loop
y = y

print(len(x))
print(len(y))

#Train and test split with random state as 64 to create static split.
x_train, x_test, y_train, y_test = train_test_split(x_loop, y, test_size=0.2, random_state=64)

print(len(y_test))
print(len(y_train))
print(len(x_test))
print(len(x_train))

#Reshape the training data to 9X20X1 for convolutional neural network
x_train_reshape = []
for j in x_train:
    x_train_reshape.append(np.reshape(j, (9, 20, 1)))

print("reshape count" + str(len(x_train_reshape)))
print(x_train_reshape[0])

#Reshape the test data to 9X20X1 for convolutional neural network
x_test_reshape = []
for k in x_test:
    x_test_reshape.append(np.reshape(k, (9, 20, 1)))

# Converting categorical data to categorical
num_categories = 3
y_train = keras.utils.to_categorical(y_train, num_categories)
y_test = keras.utils.to_categorical(y_test, num_categories)

# Model Building CNN (Convolutional Neural Network) using tensorflow and keras
num_filters = 8
filter_size = 3
pool_size = 2

modelcnn = keras.models.Sequential()
#Input
modelcnn.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(9, 20, 1)))
modelcnn.add(keras.layers.Dropout(0.25))
#Dense layer 1
modelcnn.add(keras.layers.Dense((180, activation='relu'))
modelcnn.add(keras.layers.Dropout(0.3))
#Dense layer 2
modelcnn.add(keras.layers.Dense(90, activation='relu'))
modelcnn.add(keras.layers.Dropout(0.3))
#Flatten of the input for the output layer
modelcnn.add(keras.layers.Flatten())
modelcnn.add(keras.layers.Dense(3, activation='softmax'))



# Compiling the model - adaDelta - Adaptive learning
modelcnn.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

#Print model summary
print(modelcnn.summary())
# Training and evaluating
batch_size = 50
num_epoch = 10
#Model fit
model_log = modelcnn.fit(np.array(x_train_reshape), np.array(y_train), batch_size=batch_size, epochs=num_epoch, verbose=1,
                         validation_data=(np.array(x_test_reshape), np.array(y_test)), callbacks=[PlotLossesKeras()])

#Train and Test score
train_score = modelcnn.evaluate(np.array(x_train_reshape), np.array(y_train), verbose=1)
test_score = modelcnn.evaluate(np.array(x_test_reshape), np.array(y_test), verbose=1)
print('Train accuracy CNN:', train_score[1])
print('Test accuracy CNN:', test_score[1])

#Plotting of the model training history

# list all data in history
print(model_log.history.keys())
# summarize history for accuracy
plt.plot(model_log.history['acc'])
#plt.plot(model_log.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model_log.history['loss'])
#plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Model Building ANN (Artificial Neural Network)
model = keras.models.Sequential()
model.add(keras.layers.Dense(180, activation="relu", input_shape=(180,)))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(90, activation="relu"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(3, activation="softmax"))


# Compiling the model - adaDelta - Adaptive learning
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

#Print model summary
print(model.summary())
# Training and evaluating
batch_size = 50
num_epoch = 10

model_log_ann = model.fit(np.array(x_train), np.array(y_train),
                       batch_size=batch_size, epochs=num_epoch, verbose=1,
                          validation_data=(np.array(x_test), np.array(y_test)), callbacks=[PlotLossesKeras()])

train_score = model.evaluate(np.array(x_train), np.array(y_train), verbose=1)
test_score = model.evaluate(np.array(x_test), np.array(y_test), verbose=1)
print('ANN Train accuracy:', train_score[1])
print('ANN Test accuracy:', test_score[1])

# list all data in history
print(model_log_ann.history.keys())
# summarize history for accuracy
plt.plot(model_log_ann.history['acc'])
#plt.plot(model_log_ann.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model_log_ann.history['loss'])
#plt.plot(model_log_ann.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

