import tensorflow as tf
'''
NaiveDense represents a single Dense Layer in the Keras API
Implements the following input transformation, where W and b are model parameters that can be learned
and activation is an element-wise function (usually relu)
output = activation(dot(W, input) + b)
'''

class NaiveDense :
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        # Create a matrix W of size (input_size, output_size) with random initial values
        w_shape = (input_size, output_size)
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value)

        # Create a vector b of shape (output_size,), initialized with zeroes
        b_shape = (output_size,)
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    # Apply the calculation (forward pass)
    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    # retrieve the layer's weights
    @property
    def weights(self):
        return [self.W, self.b]