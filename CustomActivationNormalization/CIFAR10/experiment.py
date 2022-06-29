"""
Evaluate the general performance and adversarial robustness of a binary CIFAR10 CNN-based classifier with fully
connected output layers which employ a novel activation function.
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers
import helper as hlp
import numpy as np
import models as my_models
import layers as my_layers
import attacks as atks

# fixes CUDNN_STATUS_INTERNAL_ERROR
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# region data preparation ####################################################
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

train_labels_one_hot = hlp.convert_to_one_hot(train_labels)
test_labels_one_hot = hlp.convert_to_one_hot(test_labels)

# class 0: automobile; class 1: horse
train_dataset = hlp.filter_and_convert_to_binary(train_images, train_labels_one_hot, 1, 7)
test_dataset = hlp.filter_and_convert_to_binary(test_images, test_labels_one_hot, 1, 7)

train_images = train_dataset[0]
train_labels = train_dataset[1]
test_images = test_dataset[0]
test_labels = test_dataset[1]
# endregion #######################################################

# region model definition ####################################################
alpha = 1.0
l2 = 0.0000
# epochs = 100
epochs = 25
batch_size = 64

inputs = tf.keras.Input(shape=(32, 32, 3))
conv1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2))(inputs)
mp1 = layers.MaxPooling2D((2, 2))(conv1)
conv2 = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2))(mp1)
mp2 = layers.MaxPooling2D((2, 2))(conv2)
conv3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2))(mp2)

conv3_flattened = layers.Flatten()(conv3)

fc1 = my_layers.CustomNormalizedDense(64, activation='relu', alpha=alpha)(conv3_flattened)
fc2 = my_layers.CustomNormalizedDense(64, activation='relu', alpha=alpha)(fc1)

# use standard fully layers for comparison:
# fc1 = layers.Dense(64, activation='relu')(conv3_flattened)
# fc2 = layers.Dense(64, activation='relu')(fc1)

outputs = my_layers.CustomNormalizedDense(1, activation=tf.nn.sigmoid, alpha=alpha)(fc2)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.BinaryAccuracy()],
              run_eagerly=True)
# endregion #######################################################

# region training & evaluation ####################################################
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_binary_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                    validation_data=(test_images, test_labels), callbacks=[model_checkpoint_callback])

model.load_weights(checkpoint_filepath)

_, test_acc = model.evaluate(test_images, test_labels, verbose=2, batch_size=32)
print(test_acc)

model = my_models.Bin_Classifier(classifier=model)
preds = model.classify(test_images)
cmat = tf.math.confusion_matrix(test_labels.flatten(), preds)
print(cmat)
# endregion #######################################################

# region crafting adversarial examples  ##################################################
params = {
    'norm': '2',
    'diff_decision_boundary': 0.01,
    'step_size': 0.001,
    'step_size_increment': 0.000001,
    'momentum': 0.9,
    'd_penalty': 0.001,
    # 'iter': 1000000,
    'iter': 10000,
    'n_candidates_beacon': 10,
    'step_size_beacon': 0.1,
    'dtype': tf.float32,
    'display_step': 1000,
}
print('params', params)

xs = np.concatenate((test_images[:100], test_images[-100:]))
ys = np.concatenate((test_labels[:100], test_labels[-100:]))

x_adv = atks.adversarial(xs, ys, model, **params)

atks.print_adversarial_ds(xs, x_adv, ys, norm='2')

hlp.show_img_grid(xs[:25], 'clean samples of cars')
hlp.show_img_grid(x_adv[:25], 'adversarial examples of cars')
hlp.show_img_grid(xs[100:125], 'clean samples of horses')
hlp.show_img_grid(x_adv[100:125], 'adversarial examples of horses')
# endregion #######################################################