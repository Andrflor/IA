from functions import *

# This is a more complex perceptron
# It is scalable for multilayer
# It uses linear algebra, bernouilli probability and gradient descent
class Perceptron:

    def __init__(self, func=sig, step=0.15):
        self.intiated = false
        self.func = func
        self.step = step
        self.b = 0

    def compute(self, X):
        self.intput = X

        if !self.intiated:
            self.w = xavier(1, self.input.shape)

        self.output = self.func(zf(self.w, self.b, X))
        return self.output

    def get_loss(self, Y):
        self.realOut = Y
        self.loss = lf(self.output, Y)
        return self.loss

    def grad(self, Y, wb):
        return dlf(self.output, Y, wb)

    def updateW(self):
        self.w-=self.grad(self.realOut, self.input)

    def updateB(self):
        self.b-=self.grad(self.realOut, 1)

    def classify(self, X):
        return np.round(self.compute(X))

    def accuracy(self, X, Y):
        return 100*np.sum(np.abs(self.classify(X)+Y-1))/X.shape

    def train(self, X, Y):
        
