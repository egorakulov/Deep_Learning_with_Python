# Wraps a list of layers
# Equivalent to Sequential in the Keras API
# This is the model you pass into other functions

class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    # calls the individual layers on the inputs
    # Forward pass through the whole model
    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights