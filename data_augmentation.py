import numpy as np
from numpy.random import default_rng
import helper as hlp


# move d distance in the direction of opposite class samples
def get_aug_bubble(xs, ys, d, n_candidates, random_seed=None):
    def get_beacon(x, beacon_pool, n_candidates):
        idcs_rand = rng.randint(0, len(xs_cls1), n_candidates)
        xs_beacon = beacon_pool[idcs_rand]
        ds = np.linalg.norm(xs_beacon - x, 2, axis=1)
        idcs_sort_ds = np.argsort(ds)

        return beacon_pool[idcs_sort_ds[0]], ds[idcs_sort_ds[0]]

    rng = np.random.RandomState(random_seed)

    xs_aug = np.zeros(xs.shape)
    ys_aug = np.zeros(ys.shape)
    xs_cls0, _ = hlp.filter_data_by_class(xs, ys, 0)
    xs_cls1, _ = hlp.filter_data_by_class(xs, ys, 1)
    for i in range(len(xs)):
        x = xs[i]
        if ys[i] == 0:
            x_beacon, d_beacon = get_beacon(x, xs_cls1, n_candidates)
        else:
            x_beacon, d_beacon = get_beacon(x, xs_cls0, n_candidates)

        d_ = min(d/2, d_beacon)
        direction_vec = (x_beacon - x) / np.linalg.norm(x_beacon - x, 2)
        xs_aug[i] = x + d_ * direction_vec
        ys_aug[i] = ys[i]

    xs_res = np.concatenate((xs, xs_aug))
    ys_res = np.concatenate((ys, ys_aug))

    return xs_res, ys_res