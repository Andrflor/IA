import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.datasets import mnist
from math import exp

#Use https://intoli.com/blog/neural-network-initialization/ to setup base weights

def relu(x):
    if x<0:
        return 0
    return x

def sigmoid(x):
    if x< -10: return 0
    return 1/(1+exp(-x))

class Perceptron:
    def __init__(self, func=relu):
        self.bias=0
        self.weights=[]
        self.func=func

    def compute(self, matrix):
        if self.weights==[]:
            for i in range(len(matrix)):
                self.weights.append(0)

        if len(self.weights)!=len(matrix):
            raise ValueError("Matrix and weights must have same length")

        res = 0
        for i in range(len(self.weights)):
            res += self.weights[i]*matrix[i]

        return self.func(res+self.bias)

class Layer:
    def __init__(self, elements, func=relu, parent=None):
        self.perceptons = []
        for i in range(elements):
            self.perceptons.append(Perceptron(func=func))

    def compute(self,matrix):
        result = []
        for perceptron in self.perceptons:
            result.append(perceptron.compute(matrix))

        return result

class MLP:
    def __init__(self, layers, num_class, batch=100, epoch=5, lr=0.001):
        self.layers=[]
        if not isinstance(layers, list):
            raise ValueError("Layers must be formatted in a list")

        for layer in layers:
            self.layers.append(Layer(layer))

        self.layers.append(Layer(num_class, func=sigmoid))

    def compute(self, matrix):
        to_input=matrix
        for layer in self.layers:
            to_input=layer.compute(to_input)

        return to_input

    def classify(self, matrix):
        computation=self.compute(matrix)
        maximum=0
        selected=-1
        for i in range(len(computation)):
            if computation[i]>maximum:
                maximum=computation[i]
                selected=i
        return selected

def load_data():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    print("Loading Minst Handwriting numbers from library")
    print("---------------")
    print("Train: x=%s, y=%s" % (trainX.shape, trainY.shape))
    print("Test: x=%s, y=%s" % (testX.shape, testY.shape))
    print("---------------")
    flatTrain = []
    flatTest = []
    for element in trainX:
        flatTrain.append(element.flatten())
        flatTest.append(element.flatten())
    return flatTrain, trainY, flatTest, testY

if __name__ == "__main__":
    trainX, trainY, TestX, TestY = load_data()
    print(trainY[0])
    mlp = MLP([len(trainX[0]), 500], 10)
    print(mlp.compute(trainX[0]))
    print(mlp.classify(trainX[0]))
