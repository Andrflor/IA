import numpy as np

#Fonction a=sigmoid(z)=1/(1+e^-z)
def sig(z):
    return 1/(1+np.exp(-z))

#Fonction z=Sigma(Wi*Xi)+b
def zf(W, b, X):
    return np.sum(W*X)+b

#Fonction LogLoss -1/m*Sigma_m(yi*log(ai)+(1-yi)*log(1-ai))
def lf(A,Y):
    l=np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    return -l

#LogLoss partial derivative xi/m*Sigma(ai-yi)
def dlf(A, Y, X):
    m = A.shape
    dl=A-Y
    return (dl*X)/m

#Xavier weight initialisation
#N is the number of perceptron in a given layer
def xavier(n, size):
    return np.random.normal(0, 1/np.sqrt(n) , size)
