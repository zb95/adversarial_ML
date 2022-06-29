import numpy as np
import matplotlib.pyplot as plt
import math


def show_img_grid(imgs, title=''):
    sqrt = math.sqrt(len(imgs))
    if not sqrt.is_integer():
        raise Exception('Array length must be a perfect square!')
    plt.figure(figsize=(10, 10))
    for i in range(len(imgs)):
        plt.subplot(sqrt, sqrt, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imgs[i])
    plt.suptitle(title)
    plt.show()


# visualize weights as a heatmap
def heatmap(W, background='white', title=''):
    len_side = int(math.ceil(math.sqrt(W.size)))
    remainder =  (len_side * len_side) - W.size
    if len(W.shape) == 2 and W.shape[1] == 1:
        padding = np.zeros((remainder, 1))
    elif len(W.shape) == 1:
        padding = np.zeros(remainder)
    else:
        raise Exception('W has invalid shape: {}'.format(W.shape))
    W_padded = np.concatenate((W, padding))
    norm_term = max(abs(W))
    norm_term = norm_term if norm_term > 0 else 1
    W_n = W_padded / norm_term

    if background == 'white':
        rgb_img = np.reshape([[1, 1 - w, 1 - w] if w >= 0 else [1 - abs(w), 1, 1- abs(w)] for w in W_n], (len_side, len_side, 3)).astype(np.float32)
    else:
        rgb_img = np.reshape([[w, 0., 0.] if w >= 0 else [0., abs(w), 0.] for w in W_n], (len_side, len_side, 3)).astype(np.float32)

    plt.title(title)
    plt.imshow(rgb_img)
    plt.tight_layout()
    plt.show()


def get_random_indices(start, end, count):
    idcs = np.arange(start, end)
    np.random.shuffle(idcs)
    return idcs[:count]


def get_batch(batch_size, data, labels, random_state=None):
    idx = np.arange(0 , len(data))
    if random_state is not None:
        random_state.shuffle(idx)
    else:
        np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def split_dataset(pct_tst, data, random_seed=None):
    idcs = np.arange(0, len(data))
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(idcs)
    tst_size = int(pct_tst * len(data))
    idcs_tst = idcs[:tst_size]
    idcs_tr = idcs[tst_size:]
    data_tst = data[idcs_tst]
    data_tr = data[idcs_tr]

    return {'data_tr': data_tr, 'data_tst': data_tst}


def get_augmented_batch(xs, ys, f_aug, n_input_samples):
    xs_batch, ys_batch = get_batch(n_input_samples, xs, ys)
    return f_aug(xs_batch, ys_batch)


def filter_data_by_class(X, Y, cls):
    Y_shape = Y.shape
    Y = Y.flatten()
    idcs = [i for i in range(len(Y)) if Y[i] == cls]

    X_filtered = np.array([X[idx] for idx in idcs])
    Y_filtered = np.array([Y[idx] for idx in idcs])

    if len(Y_shape) == 2:
        Y_filtered = np.reshape(Y_filtered, (len(Y_filtered), Y_shape[1]))

    return X_filtered, Y_filtered


def convert_to_one_hot(ys):
    ys_one_hot = np.zeros((len(ys), 10))
    ys_one_hot[range(len(ys)), ys.flatten()] = 1
    return ys_one_hot


def filter_and_convert_to_binary(data, labels, class0, class1):
    def find_indices(lst, elem):
        return [i for i, x in enumerate(lst) if np.array_equal(x, elem)]

    def filter_and_get_new_labels(old_class, new_class):
        target = [0.] * labels[0].size
        target[old_class] = 1
        indices = find_indices(labels, np.asarray(target))
        filtered_data = [data[i] for i in indices]
        filtered_labels = [[new_class]] * len(indices)
        return (filtered_data, filtered_labels)


    (data_cls0, labels_cls0) = filter_and_get_new_labels(class0, 0)
    (data_cls1, labels_cls1) = filter_and_get_new_labels(class1, 1)

    new_data = np.concatenate((data_cls0, data_cls1))
    new_labels = np.concatenate((labels_cls0, labels_cls1))

    return(new_data, new_labels)


def plot_boundary_lr(w1, w2, b, x_lim, y_lim, label='', color=None):
    if w2 == 0:
        x0 = -b / w1
        plt.plot([x0, x0], [y_lim[0], y_lim[1]], label=label, color=color)
    else:
        get_x1 = lambda x0: (-(w1 * x0) - b) / w2
        plt.plot([x_lim[0], x_lim[1]], [get_x1(x_lim[0]), get_x1(x_lim[1])], label=label, color=color)


# only works for a DNN with a single hidden ReLU layer
def plot_boundary_relu(model, x_lim1, x_lim2, n_points):
    def relu(x):
        return max(x, 0)
    def get_x2(x1):
        w_1_11 = float(model.layers[0]['W'][0][0])
        w_1_12 = float(model.layers[0]['W'][0][1])
        b_1_1 = float(model.layers[0]['b'][0])
        w_1_21 = float(model.layers[0]['W'][1][0])
        w_1_22 = float(model.layers[0]['W'][1][1])
        b_1_2 = float(model.layers[0]['b'][1])
        w_2_1 = float(model.layers[1]['W'][0][0])
        w_2_2 = float(model.layers[1]['W'][1][0])
        b_2 = float(model.layers[1]['b'][0])

        x2 = (-x1 * w_1_11 * w_2_1 - b_1_1 * w_2_1 - x1 * w_1_12 * w_2_2 - b_1_2 * w_2_2 - b_2) / (
                    w_1_21 * w_2_1 + w_1_22 * w_2_2)

        if (x1*w_1_11 + x2*w_1_21 + b_1_1 > 0) and (x1*w_1_12 + x2*w_1_22 + b_1_2 > 0):
            d = relu(x1 * w_1_11 + x2 * w_1_21 + b_1_1) * w_2_1 + relu(x1 * w_1_12 + x2 * w_1_22 + b_1_2) * w_2_2 + b_2
            if abs(d) > 1000000:
                print('meow')
            print('d', d)
            return x2
        # TODO bug: x2 needs to be reevaluated before checking the condition
        elif x1*w_1_11 + x2*w_1_21 + b_1_1 > 0:
            x2 = (-x1*w_1_11*w_2_1 -b_1_1*w_2_1 -b_2) / (w_1_21*w_2_1)
        elif x1*w_1_12 + x2*w_1_22 + b_1_2 > 0:
            x2 = (-x1*w_1_12*w_2_2 -b_1_2*w_2_2 -b_2) / (w_1_22*w_2_2)
        else:
            x2 = -999999

        #hyperplane equation
        d = relu(x1*w_1_11 + x2*w_1_21 + b_1_1 ) * w_2_1 + relu(x1*w_1_12 + x2*w_1_22 + b_1_2 ) * w_2_2 + b_2
        if abs(d) > 1000000:
            print('meow')
        print('d', d)

        return x2

    x1s = np.linspace(x_lim1, x_lim2, n_points)
    x2s = [get_x2(x1) for x1 in x1s]
    plt.scatter(x1s, x2s)


def plot_boundary_search(model, x_lim1, x_lim2, n_points_x, y_lim1, y_lim2, n_points_search, iter=5, epsilon=1e-3):
    def get_x2(x1):
        y_lim1_ = y_lim1
        y_lim2_ = y_lim2
        x2 = -999999
        for i in range(0, iter):
            x2s = np.linspace(y_lim1_, y_lim2_, n_points_search)
            candidates = np.array([[x1, x2] for x2 in x2s])
            preds = model(candidates).numpy().flatten()
            diffs = np.abs(preds - 0.5)
            idx = np.argmin(diffs)
            x2 = candidates[idx][1]
            if idx > 0:
                y_lim1_ = x2s[idx-1]
            if idx < n_points_search-1:
                y_lim2_ = x2s[idx+1]
        diff_pred = abs(float(model(np.array([[x1, x2]]))) - 0.5)
        if diff_pred < epsilon:
            return x2
        else:
            return -999999

    x1s = np.linspace(x_lim1, x_lim2, n_points_x)
    x2s = [get_x2(x1) for x1 in x1s]
    plt.scatter(x1s, x2s)