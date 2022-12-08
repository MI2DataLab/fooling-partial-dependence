# ---------------------------------
# test the algorithm implementation
# on a multidimensional XOR problem
# > python xor.py
# ---------------------------------

import os
import tensorflow as tf
from argparse import ArgumentParser, Namespace
import code
import numpy as np
import pandas as pd


def arguments() -> Namespace:
    parser = ArgumentParser(description="main")
    parser.add_argument(
        "--algorithm", default="gradient", type=str, help="algorithm type"
    )
    parser.add_argument("--strategy", default="target", type=str, help="strategy type")
    parser.add_argument("--center", default=None, type=str, help="center in strategy r")
    parser.add_argument("--n", default=320, type=int, help="number of observations")
    parser.add_argument("--dim", default=3, type=int, help="number of dimensions of X")
    parser.add_argument(
        "--size", default=16, type=int, help="number of neurons in layers"
    )
    parser.add_argument("--iter", default=50, type=int, help="max iterations")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument(
        "--lr", default=1e-2, type=float, help="learning rate for gradient algorithm"
    )
    parser.add_argument(
        "--constrain",
        default=False,
        type=bool,
        help="choose wether to constrain data or not",
    )
    parser.add_argument("--variable", default="x1", type=str, help="variable")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arguments()
    tf.random.set_seed(args.seed)

    # Friedman dataset as in https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html
    np.random.seed(args.seed)
    X = np.random.uniform(size=(args.n, 5))
    X[:, args.dim :] = 0.5
    y = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )
    EXPECTED_Y_VALUE = 14.5
    y = (y > EXPECTED_Y_VALUE).astype(int)
    X = X[:, : args.dim]
    X = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(args.dim)])

    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(X)

    model = tf.keras.Sequential()
    model.add(normalizer)
    model.add(tf.keras.layers.Dense(args.size, activation="relu"))
    model.add(tf.keras.layers.Dense(args.size, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.SGD(),
        metrics=["acc", "AUC"],
    )
    model.fit(X, y, batch_size=int(args.n / 10), epochs=300, verbose=0)

    explainer = code.Explainer(model, X, constrain=args.constrain)

    if args.algorithm == "gradient":
        alg = code.GradientAlgorithm(explainer, variable=args.variable, learning_rate=args.lr)
    else:
        alg = code.GeneticAlgorithm(explainer, variable=args.variable, std_ratio=1 / 6)

    if args.strategy == "target":
        alg.fool_aim(max_iter=args.iter, random_state=args.seed)
    else:
        alg.fool(max_iter=args.iter, center=args.center, random_state=args.seed)

    BASE_DIR = f"imgs/friedman/{args.size}_{args.n}_{args.seed}_{args.algorithm}_{args.lr}_{args.iter}_{args.variable}"
    if args.constrain:
        BASE_DIR += "_constrained"
    os.makedirs(BASE_DIR, exist_ok=True)

    alg.plot_losses(savefig=f"{BASE_DIR}/loss")
    alg.plot_explanation(savefig=f"{BASE_DIR}/expl")
    alg.plot_other_explanation(savefig=f"{BASE_DIR}/expl")
    alg.plot_data(constant=False, savefig=BASE_DIR)
    alg.get_metrics(f"{BASE_DIR}/metrics.txt")
