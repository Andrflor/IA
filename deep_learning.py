import numpy as np
import random as random

# Disable any overflow warning
np.seterr(all='warn')

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Prime derivative of tanh
def prime_tanh(x):
    return 4/(np.exp(x)+np.exp(-x))**2

class DeepLearning:

    def __init__(self, insize, l1size, l2size, l3size):

        # Init network shapes
        self.insize = insize
        self.l1size = l1size
        self.l2size = l2size
        self.l3size = l3size

        # Learning rate
        self.alpha = 0.01
        self.decay_rate = 0.001

        # Init weight matrices with variance of 1/n[l-1]
        self.w1 = np.random.randn(l1size, insize)*np.sqrt(1/insize)
        self.w2 = np.random.randn(l2size, l1size)*np.sqrt(1/l1size)
        self.w3 = np.random.randn(l3size, l2size)*np.sqrt(1/l2size)

        # Init vector biases
        self.b1 = np.zeros((l1size, 1))
        self.b2 = np.zeros((l2size, 1))
        self.b3 = np.zeros((l3size, 1))


    # Fit data to compute prediction
    def fit(self, x):

        # Calculate layer 1 output
        self.z1 = np.matmul(self.w1,x) + self.b1
        self.a1 = np.tanh(self.z1)

        # Calculate hidden layer output
        self.z2 = np.matmul(self.w2, self.a1) + self.b2
        self.a2 = np.tanh(self.z2)

        # Calculate ouput layer
        self.z3 = np.matmul(self.w3, self.a2) + self.b3
        self.a3 = sigmoid(self.z3)

        return self.a3

    # Make the actual prediction
    def predict(self, x):

        return np.round(self.fit(x))

    # Train the algorithm
    def train(self, x, y, epoch=0):

        # Coeff for training by batch
        c = 1/x.shape[0]

        # Calcul the actual model prediction
        self.fit(x)

        # Derivate the loss fonction
        dz3 = self.a3 - y
        dw3 = c*np.matmul(dz3, np.transpose(self.a2))
        db3 = c*np.sum(dw3, axis=1, keepdims=True)


        # Backpropagating to second layer
        dz2 = np.matmul(np.transpose(self.w3), dz3, prime_tanh(self.z2))
        dw2 = c*np.matmul(dz2, np.transpose(self.a1))
        db2 = c*np.sum(dz2, axis=1, keepdims=True)

        # Backpropagating to first layer
        dz1 = np.matmul(np.transpose(self.w2), dz2, prime_tanh(self.z1))
        dw1 = c*np.matmul(dz1, np.transpose(x))
        db1 = c*np.sum(dz1, axis=1, keepdims=True)

        alpha = self.alpha/(1+self.decay_rate*epoch)

        # Updating weights
        self.w1 -= alpha*dw1
        self.w2 -= alpha*dw2
        self.w3 -= alpha*dw3

        # Updating biases
        self.b1 -= alpha*db1
        self.b2 -= alpha*db2
        self.b3 -= alpha*db3

    def train_all(self, X, Y, epoch=300, batch=10):
        print("Training model for %i epochs" % epoch)
        for i in range(epoch):
            self.train(X, Y, i)
        print("Training model ended")

    def accuracy(self, X, Y):

        acc = 100*(1-np.sum(np.absolute(self.predict(X) - Y))/Y.shape[1])
        print("Model accuracy is: %.01f percents" % acc)

        return acc

# Batch normalisation
def normalize(vect, axis=1):
    var = np.sqrt(np.var(vect, axis=axis))+0.00000001
    means = np.sum(vect, axis=axis)/vect.shape[axis]

    if axis == 1:
        vect = vect - np.broadcast_to(np.transpose([means]), vect.shape)
        vect = vect/np.broadcast_to(np.transpose([var]), vect.shape)

    else:
        vect = vect - np.broadcast_to([means], vect.shape)
        vect = vect/np.broadcast_to([var], vect.shape)


    return vect

# Load any formatted dataset
def train_test_split(filename):
    XY = []
    with open(filename) as f:
        data = f.readlines()
    for i in range(len(data)):
        info = [float(point) for point in data[i].split(",")]
        XY.append(info)

    # Random shuffle for the train test split with a seed to get same results
    random.Random(0).shuffle(XY)
    X = np.transpose(np.array(XY)[:,:-1])

    # Center and reduce
    X = normalize(X)

    Y = np.transpose(np.array(XY)[:, -1:])

    # Split in 1/3 - 2/3
    split = X.shape[1]//3
    X_test = X[:, :split]
    X_train = X[:, split:]
    Y_test = Y[:, :split]
    Y_train = Y[:, split:]

    return X_train, Y_train, X_test, Y_test

# Test against specific dataset
def test_algo(filename, l1, l2, epoch=300, batch=1):

    print("\nTesting algorithm for:", filename)

    # Get a split for the data with good format
    X_train, Y_train, X_test, Y_test = train_test_split(filename)

    # Initialising model with correct size
    algo = DeepLearning(X_train.shape[0], l1, l2, Y_train.shape[0])

    # Test basic accuracy, should be around 50%
    first_run = algo.accuracy(X_test, Y_test)

    # Actually train the model
    algo.train_all(X_train, Y_train, epoch, batch)

    # Accuracy should increase significantly
    second_run = algo.accuracy(X_test, Y_test)

    # Display difference
    print("Model increased performace by %.02f percents" % (second_run-first_run))

if __name__ == '__main__':
    # The result should be more than 75%
    test_algo("diabetes", 7, 7, 2000)
    # The result should be around 90%
    test_algo("ionosphere", 7, 7, 2000)
    # The result should be more than 75%
    test_algo("titanic", 7, 7, 2000)
