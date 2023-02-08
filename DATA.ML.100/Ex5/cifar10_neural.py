import pickle
import numpy as np
import random as rand
from matplotlib import pyplot as plt

import tensorflow
from tensorflow.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

import keras

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def class_acc(pred,gt):
    n = len(gt)
    correct = 0
    for i in range(0, n):
        if pred[i] == gt[i]:
            correct += 1
    print(correct)
    return correct/n*100

def cifar10_classifier_random(x):
    n = len(x)
    random_class = []
    for i in range(0,n):
        random_class.append(rand.randint(0,9))
    return random_class


# Creates training data and labels from the batches
t1 = unpickle('../Ex5/cifar-10-batches-py/data_batch_1')
t2 = unpickle('../Ex5/cifar-10-batches-py/data_batch_2')
t3 = unpickle('../Ex5/cifar-10-batches-py/data_batch_3')
t4 = unpickle('../Ex5/cifar-10-batches-py/data_batch_4')
t5 = unpickle('../Ex5/cifar-10-batches-py/data_batch_5')

# Combines the batches
data = np.concatenate((t1["data"],t2["data"],t3["data"],t4["data"],t5["data"]))
label = np.concatenate((t1["labels"],t2["labels"],t3["labels"],t4["labels"],t5["labels"]))

# Reshape data and scale it to range [0,1]
data = data.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
tr_label = np.array(label)

# change labels to one hot vector (apparently not needed for this)
one_hot_train = np.zeros(shape=(tr_label.size,10))
one_hot_train[np.arange(tr_label.size), tr_label] = 1

'''
# Sub set for faster testing
data = data[:500]
tr_label = tr_label[:500]
'''

# Load test data
datadict = unpickle(r'../Ex5/cifar-10-batches-py/test_batch')
test_data = datadict["data"]
test_label = datadict["labels"]

test_data = test_data.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
test_label = np.array(test_label)

# change labels to one hot vector (apparently not needed for this)
one_hot_test = np.zeros(shape=(test_label.size,10))
one_hot_test[np.arange(test_label.size), test_label] = 1

'''
# Sub set for testing
test_data = test_data[:100]
test_label = test_label[:100]
'''

# NEURAL NETWORK STUFF

# Model sequential
model = Sequential([
    layers.Rescaling(1./255,input_shape=(32,32,3)),
    layers.Conv2D(16, 3, padding='same', activation="sigmoid"),
    layers.Flatten(),
    layers.Dense(10, activation='sigmoid')
])

# Compiles the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
tr_hist = model.fit(data,tr_label, epochs=10, verbose = 1)

# test the model
test_loss, test_acc = model.evaluate(test_data,  test_label)

print('\nTest accuracy:', test_acc)