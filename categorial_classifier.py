from sklearn import preprocessing as pr
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
import numpy as np

class CategoryClass(tf.keras.Model):

  def __init__(self):
    super(CategoryClass, self).__init__()

    self.flat = tf.keras.layers.Flatten(input_shape=(28, 28))
    self.dense1 = tf.keras.layers.Dense(64, activation='relu')
    self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    self.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  def call(self, inputs):
    x = self.flat(inputs)
    x = self.dense1(x)
    return self.dense2(x)

def test_model():

    print("\nTesting model for mnist dataset")
    (X_train, Y_train) , (X_test, Y_test) = mnist.load_data()

    model = CategoryClass()

    model.fit(X_train, Y_train, epochs=200, verbose=1)
    score, acc = model.evaluate(X_test, Y_test, verbose=1)

    print('Test accuracy:', acc)

if __name__ == '__main__':
    # 95% accuracy
    test_model()
