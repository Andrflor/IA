import os

# Remove all the tensorflow annoying messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn import preprocessing as pr
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.vis_utils import plot_model
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.image as mpimg
from tensorflow.keras import layers, models

# Matplotlib style and backend
# Remove if using regular backend
matplotlib.use('module://matplotlib-backend-kitty')
plt.style.use("dark_background")

# Function to lead data
def load_data(filename):

    # Load dataframe and pass it into numpy array
    data = pd.read_csv(filename).to_numpy()

    # Normalize X with l2 norm
    X = data[:, :-2]
    X = pr.normalize(X, norm="l2", axis=1)

    # Extract Y matrix
    Y = data[:, -2:]

    # Make the split with shuffling
    return train_test_split(X, Y, test_size=0.2, random_state=0)

def build_model():

    # Build model using fonctional API
    input_layer = tf.keras.Input(shape=(8,))
    dense = layers.Dense(128, activation=tf.nn.relu)(input_layer)
    dense = layers.Dense(128, activation=tf.nn.relu)(dense)
    dense_2 = layers.Dense(64, activation=tf.nn.relu)(dense)
    output_y1 = layers.Dense(1, name="y1_output")(dense)
    output_y2 = layers.Dense(1, name="y2_output")(dense_2)

    # Building dual output model
    return models.Model(inputs=input_layer, outputs=[output_y1,output_y2])

# Used to plot some difference
def plot_diff(title, y, y_pred):
    plt.scatter(y, y_pred)
    plt.title(title)
    plt.xlabel("Real values")
    plt.ylabel("Predicted values")
    plt.axis("equal")
    plt.plot([0,40], [0,40])
    plt.show()

# Used to plot particular metric
def plot_metric(title, metric_name, history, ylim=5):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(history.history[metric_name], color='b', label=metric_name)
    plt.plot(history.history["val_"+metric_name], color='g', label="val_"+metric_name)
    plt.show()

# Showing model form
def show_model(model):
    plot_model(model, show_shapes=True, show_layer_names=True)
    print(model.summary())
    img = mpimg.imread("model.png")
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# Model testing
def test_model(filename):

    # Loading everything
    print("\nTesting model for file:", filename)
    X_train, X_test, Y_train, Y_test = load_data(filename)

    # Building and showing result
    model = build_model()
    show_model(model)

    input("Press any key to process training...")

    # Compiling the model
    model.compile(optimizer="adam",
        loss={"y1_output": "mse", "y2_output": "mse"},
        metrics={"y1_output": tf.keras.metrics.RootMeanSquaredError(),
                "y2_output": tf.keras.metrics.RootMeanSquaredError()})

    history = model.fit(X_train, Y_train, epochs=1000, batch_size=10, validation_data=(X_test,Y_test))
    loss, y1_loss, y2_loss, y1_rmse, y2_rmse = model.evaluate(X_test, Y_test, verbose=1)

    # Compute predictions and show graphs
    y1_pred, y2_pred = model.predict(X_test)
    plot_diff("Y1 predictions", Y_test[:, 0], y1_pred)
    plot_diff("Y2 predictions", Y_test[:, 1], y2_pred)
    plot_metric("Y1 rmse", "y1_output_root_mean_squared_error", history)
    plot_metric("Y2 rmse", "y2_output_root_mean_squared_error", history)

    print('\nTest rmse on y1:', y1_rmse, "and on y2:", y2_rmse)

if __name__ == '__main__':
    test_model("efficiency")
