'''
This is the build-it-from-scratch exercise from chapter 2 of
Deep Learning with Python by Francois Chollet.
It is intended to get the user (aka me) familiar with the
underlying math behind the machine learning model.
'''
import tensorflow as tf
from NaiveDense import NaiveDense
from NaiveSequential import NaiveSequential
from TrainingLoop import fit
import keras
import numpy as np

# Machine Learning Model using the NaiveSequential NaiveDense classes
model = NaiveSequential([
    NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])

assert len(model.weights) == 4

# Get the data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Format the data
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype("float32") / 255

# Train the model
fit(model, train_images, train_labels, epochs=10, batch_size=128)

# Evaluating the model
predictions = model(test_images)
predictions = predictions.numpy()
preidcted_labels = np.argmax(predictions, axis=1)
matches = preidcted_labels == test_labels
print(f"Accuracy: {matches.mean():.2f}")