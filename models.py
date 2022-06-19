import tensorflow as tf
import numpy as np
import pickle as pkl
from tensorflow.keras import backend as K



class DNN_Bin_Classifier:
    # dim_layers includes the dimension of the input layer
    def __init__(self, dim_layers=None, filename=''):
        initializer = tf.random_uniform_initializer(minval=-1., maxval=1.)

        assert len(dim_layers) > 0 or filename != ''

        if filename != '':
            self.load_weights(filename)
        else:
            self.layers = []
            for i in range(len(dim_layers)):
                dim_column = 1 if (i == len(dim_layers) - 1) else dim_layers[i+1]
                W = tf.Variable(initializer(shape=[dim_layers[i],dim_column], dtype=tf.float64), name="weights L" + str(i))
                b = tf.Variable(initializer(shape=[dim_column], dtype=tf.float64), name="bias L" + str(i))
                self.layers.append({'W': W, 'b': b})

    def __call__(self, x, raw=False):
        output = x
        for i in range(len(self.layers)):
            output = tf.matmul(output, self.layers[i]['W']) + self.layers[i]['b']
            if (i == len(self.layers) - 1):
                if not raw:
                    output = tf.nn.sigmoid(output)
            else:
                output = tf.nn.relu(output)
        return output

    def classify(self, x):
        return 1 * (self(x).numpy() >= 0.5)

    def weights_as_array(self):
        weights = []
        for i in range(len(self.layers)):
            weights.append(self.layers[i]['W'])
            weights.append(self.layers[i]['b'])
        return weights

    # for regularization
    def weights_without_biases(self):
        weights = []
        for i in range(len(self.layers)):
            weights.append(self.layers[i]['W'])
        # weights = tf.convert_to_tensor(weights)
        return weights

    def hidden_representation(self, x):
        output = x
        for i in range(len(self.layers)-1):
            output = tf.matmul(output, self.layers[i]['W']) + self.layers[i]['b']
            output = tf.nn.relu(output)
        return output

    def last_layer_weights(self):
        idx = len(self.layers) - 1
        return self.layers[idx]['W'], self.layers[idx]['b']

    def save_weights(self, filename):
        output = open(filename, 'wb')
        layers_output = []
        for i in range(len(self.layers)):
            layers_output.append({'W': self.layers[i]['W'].numpy(), 'b': self.layers[i]['b'].numpy()})
        pkl.dump(layers_output, output)
        output.close()

    def load_weights(self, filename):
        file = open(filename, 'rb')
        layers_input = pkl.load(file)
        self.layers= []
        for i in range(len(layers_input)):
            W = tf.Variable(layers_input[i]['W'], name="weights L" + str(i))
            b = tf.Variable(layers_input[i]['b'], name="bias L" + str(i))
            self.layers.append({'W': W, 'b': b})
        file.close()


class Bin_Classifier(tf.keras.Model):
    def __init__(self, classifier, **kwargs):
        super(Bin_Classifier, self).__init__(**kwargs)
        self.classifier = classifier

    def classify(self, x):
        return 1 * (self.classifier(x).numpy() >= 0.5)

    def call(self, x, training=None, mask=None):
        x = tf.cast(x, dtype=tf.float32)
        res = self.classifier.call(x)
        return res

    def get_output_at_layer(self, x, layer_name='', layer_idx=-1):
        assert not (layer_name == '' and layer_idx < 0)
        output_layer = None
        if layer_idx >= 0:
            output_layer = self.classifier.layers[layer_idx]
        else:
            for layer in self.classifier.layers:
                if layer.name == layer_name:
                    output_layer = layer
                    break
        if output_layer is not None:
            get_intermediate_layer_output = K.function([self.layers[0].input],
                                                    [output_layer.output])
            output = get_intermediate_layer_output(np.array([x]))[0]
            return output
        raise Exception('Couldn\'t find layer with the given name')

    def get_config(self):
        return {"classifier": self.classifier}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BubbleAugmentedModel(tf.keras.Model):
    def __init__(self, r_bubble, min_val, max_val, **kwargs):
        super(BubbleAugmentedModel, self).__init__(**kwargs)
        self.r_bubble = r_bubble
        self.min_val = min_val
        self.max_val = max_val

    def train_step(self, data):
        x, y = data
        x_ = x.numpy()
        y_ = y.numpy()

        idcs_cls0 = [i for i, y_cur in enumerate(y_) if y_cur[0] == 0]
        idcs_cls1 = [i for i, y_cur in enumerate(y_) if y_cur[0] == 1]

        idcs_beacon = [np.random.choice(idcs_cls0) if y_cur[0] == 1 else np.random.choice(idcs_cls1) for y_cur in y_]

        x_beacon = np.zeros(x_.shape)
        for i in range(len(x_)):
            x_beacon = x_[idcs_beacon]
        x_beacon = tf.convert_to_tensor(x_beacon)

        x_flat = tf.reshape(x, [int(x.shape[0]), -1])
        x_beacon_flat = tf.reshape(x_beacon, [int(x_beacon.shape[0]), -1])
        norms = tf.norm(tf.subtract(x_beacon_flat, x_flat), 2, axis=1)
        norms = tf.linalg.diag(norms)
        norms_reciprocals = tf.linalg.inv(norms)
        direction_vec = tf.matmul(norms_reciprocals, tf.subtract(x_beacon_flat, x_flat))

        x_perturbed = tf.clip_by_value(x_flat + self.r_bubble * direction_vec, clip_value_min=self.min_val, clip_value_max=self.max_val)
        x_perturbed = tf.reshape(x_perturbed, x.shape)

        len_slice = int(len(x)/2)
        x_slice = x[0 : len_slice]
        x_perturbed_slice = x_perturbed[len_slice:]
        x_concat = tf.concat([x_slice, x_perturbed_slice], 0)
        y_concat = y

        with tf.GradientTape() as tape:
            y_pred = self(x_concat, training=True)  # Forward pass
            loss = self.compiled_loss(y_concat, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


