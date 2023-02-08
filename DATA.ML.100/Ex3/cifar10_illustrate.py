import pickle
import numpy as np
import random as rand

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


def cifar10_classifier_1nn(x,trdata,trlabels):

    nx = len(x)
    nt = len(trdata)

    predict_labels = []

    #for every sample in x, checks every sample in trdata and tracks the sample with closest distance
    for i in range (0, nx):
        # Closest neighbour variable that is compared to the distance between x and training data
        closest = 100000
        # Label index variable that stores the index of a neighbour which has the closest distance
        label_index = 0
        for j in range(0, nt):
            # Calculates euclidean distance between x[i] vector and trdata[i] vector and store the result
            dist = np.sqrt(np.sum(np.square(x[i] - trdata[j])))

            if dist < closest:
                closest = dist
                label_index = j
        #adds a predicted label to a list
        predict_labels.append(trlabels[label_index])
    return predict_labels

# Load the test batch
datadict = unpickle(r'cifar-10-batches-py/test_batch')

test_data = datadict["data"]
test_label = datadict["labels"]

# Change data type to int32
test_data = test_data.astype("int32")

'''
# Smaller sub set for testing
sub_test_data = test_data[:1000]
sub_test_label = test_label[:1000]
'''

# Creates training data and labels from the batches
t1 = unpickle('cifar-10-batches-py/data_batch_1')
t2 = unpickle('cifar-10-batches-py/data_batch_2')
t3 = unpickle('cifar-10-batches-py/data_batch_3')
t4 = unpickle('cifar-10-batches-py/data_batch_4')
t5 = unpickle('cifar-10-batches-py/data_batch_5')

# Combine the batches and change type to int32 as the program needs more bits to calculate distance
tr_data = np.concatenate((t1["data"],t2["data"],t3["data"],t4["data"],t5["data"])).astype("int32")
tr_label = np.concatenate((t1["labels"],t2["labels"],t3["labels"],t4["labels"],t5["labels"]))

'''
# Smaller sub set for testing
sub_tr_data = tr_data[:5000]
sub_tr_label = tr_label[:5000]
'''
# Random classifier accuracy
random_labels = cifar10_classifier_random(test_data)
print("Classifier accuracy:", class_acc(random_labels,test_label),"%")

# 1-NN with euclidean distance accuracy
predicted = cifar10_classifier_1nn(test_data,tr_data,tr_label)
print("Classifier accuracy:", class_acc(predicted,test_label),"%")