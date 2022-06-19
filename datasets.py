import numpy as np
from numpy.random import default_rng


def generate_2d_gauss_dataset(n_tr, n_tst, mean_cls0, cov_cls0, mean_cls1, cov_cls1, random_seed, svm_labels=False):
    rng = np.random.RandomState(random_seed)

    n_cls0_tr = int(n_tr/2)
    n_cls0_tst = int(n_tst/2)
    n_cls1_tr = int(n_tr/2)
    n_cls1_tst = int(n_tst/2)

    x_cls0_tr = rng.multivariate_normal(mean_cls0, cov_cls0, n_cls0_tr)
    x_cls0_tst = rng.multivariate_normal(mean_cls0, cov_cls0, n_cls0_tst)

    y_cls0_tr = np.zeros((n_cls0_tr, 1))
    y_cls0_tst = np.zeros((n_cls0_tst, 1))
    if svm_labels:
        y_cls0_tr.fill(-1)
        y_cls0_tst.fill(-1)

    x_cls1_tr = rng.multivariate_normal(mean_cls1, cov_cls1, n_cls1_tr)
    x_cls1_tst = rng.multivariate_normal(mean_cls1, cov_cls1, n_cls1_tst)
    y_cls1_tr = np.ones((n_cls1_tr, 1))
    y_cls1_tst = np.ones((n_cls1_tst, 1))

    x_tr = np.concatenate((x_cls0_tr, x_cls1_tr))
    y_tr = np.concatenate((y_cls0_tr, y_cls1_tr))
    x_tst = np.concatenate((x_cls0_tst, x_cls1_tst))
    y_tst = np.concatenate((y_cls0_tst, y_cls1_tst))

    return x_cls0_tr, x_cls0_tst, y_cls0_tr, y_cls0_tst, x_cls1_tr, x_cls1_tst, y_cls1_tr, y_cls1_tst, x_tr, y_tr, x_tst, y_tst


def generate_ball_in_a_bowl(n_tr, n_tst, random_seed):
    rng = np.random.RandomState(random_seed)

    n_cls0_tr = int(n_tr/2)
    n_cls0_tst = int(n_tst/2)
    n_cls0_tr_side = int(n_tr/4)
    n_cls0_tst_side = int(n_tst/4)
    n_cls1_tr = int(n_tr/2)
    n_cls1_tst = int(n_tst/2)

    # class 0, the arrowhead-shaped "bowl"
    mean_cls0_left = [-1, 1]
    mean_cls0_right = [1, 1]
    cov_cls0_left = [[1, -1], [1, -1]]
    cov_cls0_right = [[1, 1], [1, 1]]

    x_cls0_tr_left = rng.multivariate_normal(mean_cls0_left, cov_cls0_left, n_cls0_tr_side)
    x_cls0_tr_right = rng.multivariate_normal(mean_cls0_right, cov_cls0_right, n_cls0_tr_side)
    x_cls0_tr = np.concatenate((x_cls0_tr_left, x_cls0_tr_right))

    x_cls0_tst_left = rng.multivariate_normal(mean_cls0_left, cov_cls0_left, n_cls0_tst_side)
    x_cls0_tst_right = rng.multivariate_normal(mean_cls0_right, cov_cls0_right, n_cls0_tst_side)
    x_cls0_tst = np.concatenate((x_cls0_tst_left, x_cls0_tst_right))

    y_cls0_tr = np.zeros((n_cls0_tr_side*2, 1))
    y_cls0_tst = np.zeros((n_cls0_tst_side*2, 1))


    # class 1, the "ball"
    mean_cls1 =  [0, 2.5]
    cov_cls1 = [[0.1, 0], [0, 0.1]]
    x_cls1_tr = rng.multivariate_normal(mean_cls1, cov_cls1, n_cls1_tr)
    x_cls1_tst = rng.multivariate_normal(mean_cls1, cov_cls1, n_cls1_tst)
    y_cls1_tr = np.ones((n_cls1_tr, 1))
    y_cls1_tst = np.ones((n_cls1_tst, 1))

    x_tr = np.concatenate((x_cls0_tr, x_cls1_tr))
    y_tr = np.concatenate((y_cls0_tr, y_cls1_tr))
    x_tst = np.concatenate((x_cls0_tst, x_cls1_tst))
    y_tst = np.concatenate((y_cls0_tst, y_cls1_tst))

    return x_cls0_tr, x_cls0_tst, y_cls0_tr, y_cls0_tst, x_cls1_tr, x_cls1_tst, y_cls1_tr, y_cls1_tst, x_tr, y_tr, x_tst, y_tst
