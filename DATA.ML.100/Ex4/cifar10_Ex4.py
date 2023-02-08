import pickle
import numpy as np
import scipy.stats
from skimage.transform import resize
from scipy.stats import norm
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def cifar10_color(X, n):
    arr = np.zeros(shape=(len(X),n,n,3))
    # Resizes the picture to nxn
    for i in range (len(X)):
        arr[i] = resize(X[i], (n,n))
    # Reshape the dimensions to a form that is easier to handle
    arr = arr.reshape(len(X),n*n*3)
    return arr
def sort_data (X, Y):
    # The function will "sort" the pictures to their own labels in a dictionary of lists
    # The program can find can all the desired labels by calling labels[x] and change it to numpy array
    # List goes from 0-9 and they represent the labels
    labels = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

    for i in range(len(X)):
        labels[Y[i]].append(X[i])
    return labels

def cifar10_naivebayes_learn(Xp, Y):
    # Features of each color in each class are stored in mu, sigma and p arrays
    mu = np.zeros(shape=(10,3))
    sigma = np.zeros(shape=(10, 3))
    p = np.zeros(shape=(10, 1))

    # Sorts the data to a dictionary which keys corresponds to the labels
    labels = sort_data(Xp,Y)

    # Goes through the dictionary of lists, turning a list into a numpy array and then calculating mu and sigma of the colors in current label
    for j in range(10):
        labelx = np.array(labels[j])

        # Calculates mu for r g b
        rmu = np.mean(labelx[:,0])
        gmu = np.mean(labelx[:,1])
        bmu = np.mean(labelx[:,2])
        mu[j]=(rmu, gmu, bmu)

        # Calculates sigma for r g b
        rvar = np.var(labelx[:,0])
        gvar = np.var(labelx[:,1])
        bvar = np.var(labelx[:,2])
        sigma[j]=(rvar,gvar,bvar)

        p[j] = len(labelx)/len(Xp)

    return mu, sigma, p

def cifar10_classifier_naivebayes(X,mu,sigma,p):
    # Results stored in a list
    results = []
    for j in range(len(X)):
        x = X[j]
        class_acc = np.zeros(shape=(10,1))
        for i in range(len(p)):

            # Calculates the normal distributions of the colors regarding x
            rnormal = norm.pdf(x[0],mu[i,0],sigma[i,0])
            gnormal = norm.pdf(x[1],mu[i,1],sigma[i,1])
            bnormal = norm.pdf(x[2],mu[i,2],sigma[i,2])

            # Class normal computation
            cnorm = rnormal*gnormal*bnormal*p[i]

            # Add the value to a list
            class_acc[i] = cnorm

        #return the predicted class which is the index of the maximum value
        max_val = max(class_acc)
        result = np.where(class_acc == max_val)

        results.append(result[0])
    return np.array(results)

def cifar10_bayes_learn(Xf,Y, fv):

    # Features of each color in each class are stored in mu, sigma and p arrays
    # The dimension of the arrays are determined by feature vector
    # 1x1 picture means feature vector is 1x1x3, 2x2 picture means feature is 2x2x3 = 12
    mu = np.zeros(shape=(10,fv))
    sigma = np.zeros(shape=(10, fv, fv))
    p = np.zeros(shape=(10, 1))

    labels = sort_data(Xf,Y)

    # Goes through the dictionary of lists, turning a list into a numpy array and then calculating mu and sigma of the colors in current label
    for j in range(10):
        labelx = np.array(labels[j])

        # Calculates covariance of X
        print(labelx.T[:1], "\n")

        rgb_covar = np.cov(labelx)
        print(rgb_covar.shape)

        sigma[j] = rgb_covar

        # Calculates the mu
        for i in range(fv):
            mu[j][i]=np.mean(labelx[:, i])

        p[j] = len(labelx)/len(Xf)

    return mu, sigma, p

def cifar10_classifier_bayes(X,mu,sigma,p):
    # Results stored in a list
    results = []
    for j in range(len(X)):
        x = X[j]

        class_acc = np.zeros(shape=(10,1))
        for i in range(len(p)):
            # Calculates the multivariate normal distributions of the colors regarding x
            c_multinorm = scipy.stats.multivariate_normal.logpdf(x,mu[i],sigma[i],allow_singular=True)
            class_acc[i] = c_multinorm*p[i]

        #return the predicted class which is the index of the maximum value
        max_val = max(class_acc)
        result = np.where(class_acc == max_val)

        results.append(result[0][0])
    return np.array(results)

# Returns accuracy of the predicted labels
def class_acc(pred,gt):
    n = len(gt)
    # Stores the correct classifications
    correct = 0
    for i in range(n):
        if pred[i] == gt[i]:
            correct += 1
    return correct/n*100

# Creates training data and labels from the batches
t1 = unpickle('cifar-10-batches-py/data_batch_1')
t2 = unpickle('cifar-10-batches-py/data_batch_2')
t3 = unpickle('cifar-10-batches-py/data_batch_3')
t4 = unpickle('cifar-10-batches-py/data_batch_4')
t5 = unpickle('cifar-10-batches-py/data_batch_5')

# Combines the batches
data = np.concatenate((t1["data"],t2["data"],t3["data"],t4["data"],t5["data"]))
label = np.concatenate((t1["labels"],t2["labels"],t3["labels"],t4["labels"],t5["labels"]))

data = data.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
tr_label = np.array(label)


# Sub set for faster testing
data = data[:500]
tr_label = label[:500]


# Load test data
datadict = unpickle(r'cifar-10-batches-py/test_batch')
test_data = datadict["data"]
test_label = datadict["labels"]

test_data = test_data.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
test_label = np.array(test_label)


# Sub set for testing
test_data = test_data[:100]
test_label = test_label[:100]

# List for predictions to be plotted later
predictions = {}

# Size of the picture, n = 1 = 1x1
n=1
#feature vector
fv=n*n*3

print("Rescaling train and test data to", n,"x",n,"...")
# Rescale training data
tr_data1x1 = cifar10_color(data, n)
test_data1x1 = cifar10_color(test_data, n)

naive_features = cifar10_naivebayes_learn(tr_data1x1,tr_label)
predicted = cifar10_classifier_naivebayes(test_data1x1,naive_features[0],naive_features[1],naive_features[2])
print("Accuracy with naive bayes is:",class_acc(predicted,test_label),"%","\n")

# Loop rescales the data to nxn form and then performs better bayesian learning to the data
m = [1]
for n in m:
    fv= n*n*3
    print("Rescaling train and test data to", n, "x", n, "...")
    tr_data_nxn = cifar10_color(data, n)
    test_data_nxn = cifar10_color(test_data, n)

    features = cifar10_bayes_learn(tr_data_nxn, tr_label, fv)
    print("Making prediction ...")
    predicted = cifar10_classifier_bayes(test_data_nxn, features[0], features[1], features[2])

    acc = round(class_acc(predicted, test_label), 2)
    print("Accuracy with ",n,"x",n, "bayes is:", acc, "%\n")

    name = str(n)+"x"+str(n)
    predictions[name] = acc

names = list(predictions.keys())
values = list(predictions.values())

plt.plot(names, values)
plt.ylabel("prediction accuracy")
plt.show()