import matplotlib.pyplot as plt
from random import random, uniform
from math import exp

def sigmoid(x):
    if x< -10: return 0
    return 1/(1+exp(-x))

class Perceptron:

    def __init__(self, func=sigmoid):
        self.step=0.15
        self.bias=random()/10
        self.weights=[]
        self.func=func

    def display(self, matrix):
        print("Compute: %.4f" % p.compute(matrix))
        print("Weights: %.4f, %.4f" % (p.weights[0], p.weights[1]))
        print("Bias: ", p.bias)

    def get_eq(self):
        return -self.weights[0]/self.weights[1], -self.bias/self.weights[1]

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

    def classify(self, matrix):
        res = self.compute(matrix)

        if res > 0.5:
            return 1
        else:
            return 0

    def train(self, matrixes, iterations=500):
        for i in range(iterations):
            for matrix in matrixes:
                attended = matrix[2]
                mat=[matrix[0], matrix[1]]
                res = self.compute(mat)
                if self.classify(mat)!=attended:
                    error = attended - res
                    self.bias+=self.step*error
                    for i in range(len(self.weights)):
                        self.weights[i]+=self.step*error*mat[i]

    def test(self, data):
        print("Data generated %.2fx %.2f"% (data[5], data[6]))
        print("---------------")
        matrix = data[0]
        self.verbose_train(matrix)
        print("---------------")
        eq=self.get_eq()
        print("Preceptron eq: %.2fx %.2f" % (eq[0], eq[1]))
        print("---------------")
        print("Wrong classsification percentage: %.2f" % self.get_err(matrix))

    def verbose_train(self, matrix):
        print("First compute result")
        self.display(matrix[0][1:])
        print("------------------")
        self.train(matrix)
        print("After training result")
        self.display(matrix[0][1:])

    def get_err(self, matrix):
        err = 0
        for mat in matrix:
            if self.classify([mat[0],mat[1]]) != mat[2]:
                err+=1
        return 100*err/len(matrix)

def generateData(n=100, arange=100, brange=40):
    inputs = []
    xb = []
    yb = []
    xr = []
    yr = []
    a=uniform(-arange, arange)
    b=uniform(-brange, brange)
    for i in range(n):
        xb.append(random())
        xr.append(random())
        yb.append(a*xb[i]+b+random()*a)
        yr.append(a*xr[i]+b-random()*a)
        inputs.append([xb[i],yb[i],1])
        inputs.append([xr[i],yr[i],0])
    return inputs, xr, yr, xb, yb, a, b


def show(data, eq):
    plt.plot(data[1], data[2], 'ro')
    plt.plot(data[3], data[4], 'co')
    plt.plot([0,1], [data[6],data[5]+data[6]], '-g')
    plt.plot([0,1], [eq[1],eq[0]+eq[1]], '-y')
    plt.show()

if __name__=="__main__":
    p = Perceptron()
    data = generateData()
    p.test(data)
    show(data, p.get_eq())
