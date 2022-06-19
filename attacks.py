import  tensorflow as tf
import numpy as np
import helper as hlp


# Novel initialization method
# For each x in xs__clean, find a sample of the opposite class ("beacon") relatively close to x.
# Then, starting from x, move toward the beacon until we are just past the decision boundary
# Empirical results have shown that the points discovered by this method are not affected by the gradient masking
# phenomenon, making it easy for gradient descent to minimize the attack budget whilst ensuring misclassification
def find_init_points_on_hp(xs_clean, ys, model, xs_adv, step_size, n_candidates_beacon=1, random_seed=None):
    def norm(xs):
        xs_flat = xs.reshape(xs.shape[0], -1)
        return np.linalg.norm(xs_flat, 2, axis=1)

    # select closest sample of the opposite class from beacon_pool out of n_candidates random candidates
    def get_beacon(x, beacon_pool, y_tgt, n_candidates):
        res = None
        x_flat = x.flatten()
        while True:
            idcs_rand = rng.randint(0, len(beacon_pool), n_candidates)
            xs_beacon = beacon_pool[idcs_rand]
            xs__beacon_flat = xs_beacon.reshape(xs_beacon.shape[0], -1)
            ds = norm(xs__beacon_flat - x_flat)
            idcs_sort_ds = np.argsort(ds)
            for i in range(len(idcs_sort_ds)):
                x_candidate = xs_beacon[idcs_sort_ds[i]]
                if int(model.classify(np.array([x_candidate]))) == y_tgt:
                    res = x_candidate
                    break
            if res is not None:
                break
        return res

    xs_clean = xs_clean.numpy()
    xs_cls0, _ = hlp.filter_data_by_class(xs_clean, ys, 0)
    xs_cls1, _ = hlp.filter_data_by_class(xs_clean, ys, 1)
    xs_hp = np.zeros(xs_clean.shape)
    rng = np.random.RandomState(random_seed)

    for i in range(len(xs_clean)):
        x = xs_clean[i]

        if ys[i] == 0:
            y_tgt = 1
            x_beacon = get_beacon(x, xs_cls1, y_tgt, n_candidates_beacon)
        else:
            y_tgt = 0
            x_beacon = get_beacon(x, xs_cls0, y_tgt, n_candidates_beacon)
        direction_vec = (x_beacon - x) / np.linalg.norm(x_beacon.flatten() - x.flatten(), 2)

        while True:
            if int(model.classify(np.array([x]))) == y_tgt:
                xs_hp[i] = x
                break
            x = x + step_size * direction_vec

    xs_adv.assign(xs_hp)


# craft adversarial samples with minimal distance from the sources
# requires samples from both classes for initialization
def adversarial(samples, labels_true, model, norm, diff_decision_boundary, d_penalty, step_size, momentum, iter,
                display_step, val_min=0, val_max=1, n_candidates_beacon=1, step_size_beacon=0.01,
                step_size_increment=0., optimizer='sgd', x_adv_initial=None, dtype=tf.float64, random_seed=None):

    x_orig = tf.constant(samples)
    x_orig_flat = tf.reshape(x_orig, [int(x_orig.shape[0]), -1])
    labels_true = labels_true.flatten()
    out_tgt = [0.5 + diff_decision_boundary if y == 0 else 0.5 - diff_decision_boundary for y in labels_true]
    out_tgt = tf.cast(tf.constant(out_tgt), dtype=dtype)

    x_adv = tf.Variable(x_orig)
    if x_adv_initial is not None:
        x_adv.assign(x_adv_initial)

    out_orig = model(x_orig).numpy().flatten()
    ones = np.ones(len(labels_true))
    labels_false = ones - np.asarray(labels_true)

    # indices of naturally misclassified samples
    idcs_natural_misclsf = np.argwhere(
        [(out_tgt[i] <= out_orig[i] <= labels_false[i]) or (out_tgt[i] >= out_orig[i] >= labels_false[i]) for i in
         range(len(out_orig))])

    if x_adv_initial is None:
        find_init_points_on_hp(x_orig, labels_true, model, x_adv, step_size_beacon, n_candidates_beacon, random_seed)

    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=step_size)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=step_size, momentum=momentum)

    step_sizes_initial = np.full(len(samples), step_size)
    step_sizes_updated = np.full(len(samples), step_size)
    step_sizes_min = np.full(len(samples), step_size_increment)
    step_scales = np.ones(x_adv.shape)

    loss = None
    loss_old = None

    for i in range(1, iter + 1):
        with tf.GradientTape() as tape:
            out = tf.reshape(model(x_adv), [-1])

            # revert naturally misclassified samples
            x_adv_val = x_adv.numpy()
            x_adv_val[idcs_natural_misclsf] = samples[idcs_natural_misclsf]
            x_adv.assign(x_adv_val)

            x_adv_flat = tf.reshape(x_adv, [int(x_adv.shape[0]), -1])

            d = tf.norm(tf.subtract(x_adv_flat, x_orig_flat), float(norm), axis=1)

            if i > 1:
                loss_old = loss

            loss_dif_pred = tf.square(out_tgt - out)
            loss_d = tf.cast(d_penalty * d, dtype=dtype)
            loss = loss_dif_pred + loss_d

            if i > 1:
                step_size_changes = step_size_increment * np.sign(loss_old - loss)
                step_sizes_updated = np.maximum(step_sizes_updated + step_size_changes, step_sizes_min)
                step_scales = np.ones(x_adv.shape)
                for k in range(len(step_scales)):
                    step_scales[k] = step_scales[k] * (step_sizes_updated[k] / step_sizes_initial[k])

            grads = tape.gradient(loss, [x_adv])
            grads = tf.multiply(grads, step_scales)

            opt.apply_gradients(zip(grads, [x_adv]))
            x_adv_clipped = tf.clip_by_value(x_adv, clip_value_min=val_min, clip_value_max=val_max)
            x_adv.assign(x_adv_clipped)

            # print current progress
            if (not np.isinf(display_step) and (i % display_step == 0)) or (i == iter) or (i == 1):
                # revert naturally misclassified samples
                x_adv_val = x_adv.numpy()
                x_adv_val[idcs_natural_misclsf] = samples[idcs_natural_misclsf]
                x_adv.assign(x_adv_val)

                x_adv_flat = tf.reshape(x_adv, [int(x_adv.shape[0]), -1])
                d = tf.norm(tf.subtract(x_adv_flat, x_orig_flat), float(norm), axis=1)
                loss_d = tf.cast(d_penalty * d, dtype=dtype)
                loss = loss_dif_pred + loss_d

                # grad is NaN for naturally misclassified samples, as d=0
                grads_clean = grads[0].numpy()
                grads_clean[np.argwhere(np.isnan(grads_clean))] = 0.
                g_norm_avg = tf.reduce_mean(tf.reduce_sum(tf.abs(grads_clean), axis=2))
                loss_avg = tf.reduce_mean(loss).numpy()
                loss_dif_pred_avg = tf.reduce_mean(loss_dif_pred).numpy()
                loss_d_avg = tf.reduce_mean(loss_d).numpy()
                d_avg = tf.reduce_mean(d).numpy()
                preds = model(x_adv).numpy().flatten()
                count_misclsf = np.count_nonzero(abs(labels_true - preds) > 0.5)
                step_scale_avg = np.mean(step_scales)
                print('iter:', i, ', avg d:', d_avg, ', avg loss:', loss_avg, ', loss_dif_pred_avg:', loss_dif_pred_avg,
                      ', loss_d_avg:', loss_d_avg,', avg g_norm:', g_norm_avg.numpy(), ', count misclassifications:', count_misclsf,
                      ', step size:', step_size, ', avg step scale:', step_scale_avg)

    x_adv_val = x_adv.numpy()

    print('indices of naturally misclassified samples', idcs_natural_misclsf)

    # naturally misclassified samples don't require crafted adversarial samples
    x_adv_val[idcs_natural_misclsf] = samples[idcs_natural_misclsf]

    x_adv.assign(x_adv_val)

    print_adversarial_ds(samples, x_adv, labels_true, norm)

    preds = model(x_adv).numpy().flatten()
    print('preds', preds)

    count_misclsf = np.count_nonzero(abs(labels_true - preds) > 0.5)
    print('count misclassifications: ' + str(count_misclsf) + '/' + str(len(labels_true)))
    idcs_failed = np.argwhere(abs(labels_true - preds) < 0.5).flatten()
    print('indices of failed attacks: ', idcs_failed)

    return x_adv_val


def print_adversarial_ds(samples, x_adv, labels_true, norm):
    x_cls0, _ = hlp.filter_data_by_class(samples, labels_true, 0)
    x_cls0 = x_cls0.reshape((len(x_cls0), -1))
    x_cls1, _ = hlp.filter_data_by_class(samples, labels_true, 1)
    x_cls1 = x_cls1.reshape((len(x_cls1), -1))
    x_adv_cls0, _ = hlp.filter_data_by_class(x_adv, labels_true, 0)
    x_adv_cls0 = x_adv_cls0.reshape((len(x_adv_cls0), -1))
    x_adv_cls1, _ = hlp.filter_data_by_class(x_adv, labels_true, 1)
    x_adv_cls1 = x_adv_cls1.reshape((len(x_adv_cls1), -1))

    x_adv_flat = tf.reshape(x_adv, [int(x_adv.shape[0]), -1])
    x_orig_flat = tf.reshape(samples, [int(samples.shape[0]), -1])
    d = tf.norm(tf.subtract(x_adv_flat, x_orig_flat), float(norm), axis=1)
    d_cls0 = tf.norm(tf.subtract(x_adv_cls0, x_cls0), float(norm), axis=1)
    d_cls1 = tf.norm(tf.subtract(x_adv_cls1, x_cls1), float(norm), axis=1)

    d_avg = tf.reduce_mean(d).numpy()
    d_cls0_avg = tf.reduce_mean(d_cls0).numpy()
    d_cls1_avg = tf.reduce_mean(d_cls1).numpy()
    print('d_avg', d_avg, 'd_cls0_avg', d_cls0_avg, 'd_cls1_avg', d_cls1_avg)