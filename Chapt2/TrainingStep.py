'''
Represents a single training step for one batch of images and labels
- Runs a forward pass
- Computes the gradient of the loss function with regard to
      the weights currently in the model
- Updates the weights
'''

import tensorflow as tf
import keras

# Model achieves accuracy of 81%
learning_rate = 1e-3
# update the weights based on the gradient of that weight
# Move it opposite to calculated slope
def update_weights(gradients, weights):
    for (g, w) in zip(gradients, weights):
        w.assign_sub(g * learning_rate)


def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        # forward pass
        predictions = model(images_batch)
        per_sample_losses = keras.losses.sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)
    # Compute the gradient of the loss with regard to weights
    gradients = tape.gradient(average_loss, model.weights)
    # update the weights
    update_weights(gradients, model.weights)
    return average_loss