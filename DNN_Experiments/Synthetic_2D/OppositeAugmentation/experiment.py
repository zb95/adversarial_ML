"""
Evaluate the effect of data augmentation (using the opposite class) on the decision boundary
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import helper as hlp
import datasets as gen
from models import DNN_Bin_Classifier
import loss_functions as lf
import data_augmentation as aug


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.cast(tf.round(y_pred), tf.int64), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


n_tr = 200
n_tst = 100

x_cls0_tr, x_cls0_tst, y_cls0_tr, y_cls0_tst, x_cls1_tr, x_cls1_tst, y_cls1_tr, y_cls1_tst, x_tr, y_tr, x_tst, y_tst = gen.generate_ball_in_a_bowl(n_tr, n_tst, 1234)

learning_rate = 0.03
momentum=0.95
l1=0.0
l2=0.0
epochs = 4000
batch_size = 200
display_step = 1000

train_data = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
train_data = train_data.shuffle(5000).repeat(epochs).batch(batch_size).prefetch(1)

optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)

d_aug = 1.5

model = DNN_Bin_Classifier([2, 20, 2])

weights_arr = model.weights_as_array()
weights_without_b = model.weights_without_biases()
weights = weights_without_b

x_batch = None
y_batch = None

for step, (x,y) in enumerate(train_data):
    if d_aug > 0:
        x, y = aug.get_aug_bubble(x.numpy(), y.numpy(), d_aug, 50)
        x = tf.constant(x)
        y = tf.constant(y)
    with tf.GradientTape(persistent=True) as t:
        pred_tr = model(x)
        loss_tr = lf.l2_xe_loss(pred_tr, y, weights, l2)

    gradients = t.gradient(loss_tr, weights_arr)

    if (step == 0) or ((step + 1) % display_step == 0):
        pred_tr = model(x)
        acc_tr = accuracy(pred_tr, y)
        pred_tst = model(x_tst)

        loss_tst = lf.l2_xe_loss(pred_tst, y_tst, weights, l2)
        acc_tst = accuracy(pred_tst, y_tst)
        print(step + 1, 'train loss:', "{:.6f}".format(float(loss_tr)), 'train acc:', "{:.6f}".format(float(acc_tr)),
              'test loss:', "{:.6f}".format(float(loss_tst)), 'test acc:', "{:.6f}".format(float(acc_tst)))
        x_batch = x
        y_batch = y

    optimizer.apply_gradients(zip(gradients, weights_arr))

pred_tr = model(x_tr)
acc_tr = accuracy(pred_tr, y_tr)
pred_tst = model(x_tst)
acc_tst = accuracy(pred_tst, y_tst)
print('train acc:', "{:.6f}".format(float(acc_tr)), 'test acc:', "{:.6f}".format(float(acc_tst)))

x_lim1 = -4
x_lim2 = 4
y_lim1 = -2
y_lim2 = 5
plt.axis((x_lim1, x_lim2, y_lim1, y_lim2))

# plot the training and test samples along with the decision boundary
plt.scatter(x_cls0_tr.T[0], x_cls0_tr.T[1])
plt.scatter(x_cls1_tr.T[0], x_cls1_tr.T[1])
plt.scatter(x_cls0_tst.T[0], x_cls0_tst.T[1])
plt.scatter(x_cls1_tst.T[0], x_cls1_tst.T[1])
plt.axis((x_lim1, x_lim2, y_lim1, y_lim2))
hlp.plot_boundary_search(model, x_lim1, x_lim2, 1000, y_lim1, y_lim2, 100, epsilon=1e-4)
plt.show()

