import numpy as np
import tensorflow as tf
import functools


def loss(original, changed, aim=False, center=True):
    if aim:  # original is target
        return ((original - changed) ** 2).mean()
    else:
        if center:
            return -(
                ((original - original.mean()) - (changed - changed.mean())) ** 2
            ).mean()
        else:
            return -((original - changed) ** 2).mean()


def loss_tf(original, changed, aim=False, center=True):
    if aim:  # original is target
        return 0.5 * tf.reduce_mean((original - changed) ** 2)
    else:
        if center:
            return -0.5 * tf.reduce_mean(
                ((original - original.mean()) - (changed - tf.reduce_mean(changed))) ** 2
            )
        else:
            return -0.5 * tf.reduce_mean((original - changed) ** 2)


def loss_pop(original, changed, aim=False, center=True):
    """
    vectorized (whole population) loss calculation
    """
    original_long = np.tile(original, (changed.shape[0], 1))
    if aim:  # original is target
        return ((original_long - changed) ** 2).mean(axis=1)
    else:
        if center:
            return -(
                (
                    (original_long - original.mean())
                    - (changed - changed.mean(axis=1, keepdims=True))
                )
                ** 2
            ).mean(axis=1)
        else:
            return -((original_long - changed) ** 2).mean(axis=1)

# def loss_ks(X_original, X_changed):
#     ret = 0
#     for column in range(X_changed.shape[1]):
#         ret += ks_distance(X_original[:, column], X_changed[:, column])

#     # assert ret == loss_ks_vectorized(X_original, X_changed)

#     return ret

def loss_ks(X_original, X_changed):
    # Stack the columns of X_original and X_changed together
    stacked_samples = tf.stack([X_original, X_changed], axis=-1)
    # Reshape the stacked array to (num_columns, 2, num_samples)
    stacked_samples = tf.transpose(stacked_samples, perm=[1, 2, 0])
    # Compute the KS distance between the stacked samples
    ks_dists = tf.map_fn(lambda x: ks_distance(x[0], x[1]), stacked_samples, dtype=tf.float32)
    # Sum the KS distances over all columns
    ret = tf.reduce_sum(ks_dists)
    print("ks", ret)

    return ret


def ks_distance(sample1, sample2):
    # sample1 and sample2 have to be 1D
    def get_cdf(grid, sample):
        cdf1 = tf.map_fn(fn=lambda t: tf.reduce_sum(tf.cast(t == sample, tf.float64)), elems=grid)
        cdf1 = tf.cast(cdf1, dtype=tf.float32)
        cdf1 = tf.cumsum(cdf1)
        cdf1 = cdf1 / sample.shape[0]

        return cdf1

    grid = tf.sort(tf.unique(tf.concat([sample1, sample2], axis=0))[0])
    cdf1 = get_cdf(grid, sample1)
    cdf2 = get_cdf(grid, sample2)
    # print("grid", grid.shape)
    # print("cdf1", cdf1.shape)
    # print("cdf2", cdf2.shape)

    ks = tf.reduce_max(tf.abs(cdf1 - cdf2)) 

    return ks
