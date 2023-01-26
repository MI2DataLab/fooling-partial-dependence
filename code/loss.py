import numpy as np
import tensorflow as tf


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
                ((original - tf.reduce_mean(original)) - (changed - tf.reduce_mean(changed))) ** 2
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

            
def loss_dist(X_original, X_changed):
    x1 = tf.sort(X_original, axis=0)
    x1 = tf.cast(x1, tf.float32)
    x2 = tf.sort(X_changed, axis=0)
    x2 = tf.cast(x2, tf.float32)
    ret = tf.reduce_mean((x1 - x2) ** 2)
    return tf.cast(ret, tf.float32)
