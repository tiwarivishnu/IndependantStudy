# -*- coding: utf-8 -*-
"""
Created on Saturday July 27 17:13:06 2018
@author: vishnu
"""
# noinspection PyUnresolvedReferences
import os, sys, math
#import display from IPython.display
import keras
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_context("notebook", font_scale=1.4)
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

'exec(%matplotlib inline)'
nums = [1, 2, 3, 4, 5]
numsnew = []
for num in nums:
    numnew = num+1
    numsnew.append(numnew)
    print(numsnew)



#Required data

data = pd.read_csv("C:/VT/Independent Study/ran_peps_netMHCpan40_predicted_A0201_reduced_cleaned_balanced.csv",
                   sep='\t')


x = data.drop(columns=["label_num", "data_type", "label_chr"])
y = data.drop(columns=["peptide", "label_chr", "data_type"])

print(x)
print(y)

#Peptide encoding

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def show_matrix(m):
     #display a matrix
     cm = sns.light_palette("seagreen", as_cmap=True)
     #display(m.style.background_gradient(cmap=cm))

def one_hot_encode(seq):
    o = list(set(codes) - set(seq))
    s = pd.DataFrame(list(seq))
    x = pd.DataFrame(np.zeros((len(seq),len(o)),dtype=int),columns=o)
    a = s[0].str.get_dummies(sep=',')
    a = a.join(x)
    a = a.sort_index(axis=1)
    #show_matrix(a)
    e = a.values.flatten()
    return e

pep = 'ALDFEQEMT'

e = one_hot_encode(pep)
print(e)

# create an object to hold loop results
print("length of x:" + str(len(x)))
x = x.values
print("x length after removing header: " + str(len(x)))
x_loop = []
for i in x:
    x_loop.append(one_hot_encode(i))
    #print(x_loop)



print("x_loop length: " + str(len(x_loop)))
x = np.asmatrix(x_loop)
y = y.as_matrix()

print(len(x))
print(len(y))


#print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=64)
#print(y_test)
#print(y_train)

# Converting categorical data to categorical
num_categories = 3
y_train = keras.utils.to_categorical(y_train, num_categories)
y_test = keras.utils.to_categorical(y_test, num_categories)

#Build the models

# Model Building
model = keras.models.Sequential()
model.add(keras.layers.Dense(50, activation="tanh", input_dim=21))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(80, activation="tanh"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(3, activation="softmax"))

# Compiling the model - adaDelta - Adaptive learning
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# Training and evaluating
batch_size = 50
num_epoch = 5
model_log = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_data=(x_test, y_test))

train_score = model.evaluate(x_train, y_train, verbose=1)
test_score = model.evaluate(x_test, y_test, verbose=1)
print('Train accuracy:', train_score[1])
print('Test accuracy:', test_score[1])
