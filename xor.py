# ---------------------------------
# test the algorithm implementation
# on a multidimensional XOR problem
# > python xor.py
# ---------------------------------


import tensorflow as tf
from argparse import ArgumentParser, Namespace
import os
import code
import numpy as np
import pandas as pd
from code.model import BasicModel


def arguments() -> Namespace:
    parser = ArgumentParser(description="main")
    parser.add_argument(
        "--algorithm", default="gradient", type=str, help="algorithm type"
    )
    parser.add_argument("--strategy", default="target", type=str, help="strategy type")
    parser.add_argument("--center", default=None, type=str, help="center in strategy r")
    parser.add_argument("--n", default=320, type=int, help="number of observations")
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
    parser.add_argument(
        "--explanations",
        default=["pd", "ale"],
        type=str,
        nargs="+",
        help="list of explanations",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arguments()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    x1 = np.random.normal(size=args.n)
    x2 = np.random.normal(size=args.n)
    x3 = np.random.normal(size=args.n)
    y = 1 * (x1 * x2 * x3 > 0)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(X)

    #model = BasicModel(args.size, normalizer)
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
        alg = code.GradientAlgorithm(
            explainer,
            variable="x1",
            learning_rate=args.lr,
            explanation_names=args.explanations,
        )
    else:
        alg = code.GeneticAlgorithm(explainer, variable="x1", std_ratio=1 / 6)

    if args.strategy == "target":
        alg.fool_aim(max_iter=args.iter, random_state=args.seed)
    else:
        alg.fool(max_iter=args.iter, center=args.center, random_state=args.seed)

    BASE_DIR = f"imgs/xor/{args.size}_{args.n}_{args.seed}_{args.algorithm}_{args.lr}_{args.iter}"
    if args.constrain:
        BASE_DIR += "_constrained"
    os.makedirs(BASE_DIR, exist_ok=True)

    alg.plot_losses(savefig=f"{BASE_DIR}/loss")
    alg.plot_explanation(savefig=f"{BASE_DIR}/expl")
    alg.plot_other_explanation(savefig=f"{BASE_DIR}/expl")
    alg.plot_data(constant=False, savefig=BASE_DIR)
    alg.get_metrics(f"{BASE_DIR}/metrics")
