import tensorflow as tf


def get_xe_losses(y_pred, y_true, epsilon=0.00001):
    return ((-y_true) * tf.math.log(tf.maximum(y_pred, epsilon))) - ((1-y_true) * tf.math.log(tf.maximum(1-y_pred, epsilon)))


def xe_loss(y_pred, y_true):
    losses = get_xe_losses(y_pred, y_true)
    return tf.reduce_mean(losses)


# W might be an array of Tensors
def l2_xe_loss(y_pred, y_true, W, l2_scale):
    losses = ((-y_true) * tf.math.log(y_pred)) - ((1-y_true) * tf.math.log(1-y_pred))
    if isinstance(W, list):
        l2_penalty = l2_scale * tf.reduce_sum([tf.nn.l2_loss(weights) for weights in W])
    else:
        l2_penalty = l2_scale * tf.nn.l2_loss(W)
    return tf.reduce_mean(losses) + l2_penalty