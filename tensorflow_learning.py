from sklearn import preprocessing as pr
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
import numpy as np

# Function to lead data
def load_data(filename):

    # Load dataframe and pass it into numpy array
    data = pd.read_csv(filename).to_numpy()

    # Normalize X with l2 norm
    X = data[:, :-1]
    X = pr.normalize(X, norm="l2", axis=1)

    # Extract Y matrix
    Y = np.transpose([data[:, -1]])

    # Make the split with shuffling
    return train_test_split(X, Y, test_size=0.33, random_state=0)



class BinaryClass(tf.keras.Model):

  def __init__(self):
    super(BinaryClass, self).__init__()
    self.dense1 = tf.keras.layers.Dense(7, activation=tf.nn.tanh)
    self.dense2 = tf.keras.layers.Dense(7, activation=tf.nn.tanh)
    self.dense3 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    self.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  def call(self, inputs):
    x = self.dense1(inputs)
    x2 = self.dense2(x)
    return self.dense3(x2)

def test_model(filename):

    print("\nTesting model for file:", filename)
    X_train, X_test, Y_train, Y_test = load_data(filename)
    model = BinaryClass()

    model.fit(X_train, Y_train, epochs=2000, verbose=0)
    score, acc = model.evaluate(X_test, Y_test, verbose=0)

    print('Test accuracy:', acc)

if __name__ == '__main__':
    test_model("diabetes")
    test_model("ionosphere")
    test_model("titanic")
