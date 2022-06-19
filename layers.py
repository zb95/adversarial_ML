import tensorflow as tf
from tensorflow import keras


# divided by sum of magnitudes of inputs
class CustomNormalizedDense(keras.layers.Layer):
    def __init__(self, units, activation=None, alpha=1., dense=None, biases=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = keras.activations.get(activation)
        self.alpha = alpha
        self.trainable = True
        self.dense = dense #TODO should be tensors
        self.biases = biases
        self.rng = tf.random.Generator.from_seed(123, alg='philox')

    def build(self, batch_input_shape):
        self.dense = self.add_weight(name="dense",
                                     shape=[batch_input_shape[-1], self.units],
                                     initializer="glorot_uniform",
                                     trainable=True)
        self.biases = self.add_weight(name="bias",
                                      shape=[self.units],
                                      initializer="glorot_uniform",
                                      trainable=True)
        super().build(batch_input_shape)

    def call(self, inputs):
        inputs_weighted = tf.matmul(inputs, self.dense)
        norm_term = tf.matmul(tf.abs(inputs), tf.abs(self.dense)) + tf.abs(self.biases)
        # norm_term = tf.reduce_sum(tf.abs(inputs_weighted), axis=0) + tf.abs(self.biases)
        norm_term_interpolated = 1 + self.alpha*(norm_term-1)
        z = (inputs_weighted + self.biases) / norm_term_interpolated
        res = self.activation(z)
        return res

    def get_config(self):
        config = {'units': self.units,
                  'activation': self.activation_name,
                  'alpha': self.alpha,
                  'dense': self.dense.numpy(),
                  'biases': self.biases.numpy(),
                  'trainable': self.trainable}
        base_config = super(CustomNormalizedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


