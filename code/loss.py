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
        return tf.reduce_mean((original - changed) ** 2)
    else:
        if center:
            return -tf.reduce_mean(
                ((original - original.mean()) - (changed - tf.reduce_mean(changed))) ** 2
            )
        else:
            return -tf.reduce_mean((original - changed) ** 2)


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
