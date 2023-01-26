import tensorflow as tf
from argparse import ArgumentParser, Namespace
import os
import code
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def arguments() -> Namespace:
    parser = ArgumentParser(description="main")
    parser.add_argument("--variable", default="age", type=str, help="variable")
    parser.add_argument("--strategy", default="target", type=str, help="strategy type")
    parser.add_argument("--iter", default=50, type=int, help="max iterations")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument(
        "--lr", default=0.1, type=float, help="learning rate for gradient algorithm"
    )
    parser.add_argument("--n", default=320, type=int, help="number of observations")

    parser.add_argument(
        "--size", default=128, type=int, help="number of neurons in layers"
    )

    parser.add_argument(
        "--dist-weight",
        default=0,
        type=float,
        help="weight of distribution distance in loss",
    )

    parser.add_argument("--name", default="heart", type=str, help="dataset name")
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


def get_dataset(name):
    # this is a series of elifs because my Python is too old for match-case
    if name == "heart":
        df = pd.read_csv("data/heart.csv")
        X, y = df.drop("target", axis=1), df.target.values
        CONSTANT = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    elif name == "xor":
        x1 = np.random.normal(size=args.n)
        x2 = np.random.normal(size=args.n)
        x3 = np.random.normal(size=args.n)
        y = 1 * (x1 * x2 * x3 > 0)
        X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        CONSTANT = []

    elif name == "bike-sharing":
        df = pd.read_csv("data/bike-sharing-day.csv").dropna()
        variables_to_drop = [
            "instant",
            "dteday",
            "season",
            "weekday",
            "mnth",
        ]
        df = df.drop(variables_to_drop, axis=1)
        CONSTANT = ["yr", "workingday", "weathersit"]
        target_fields = ["cnt", "registered", "casual"]
        X = df.drop(target_fields, axis=1)
        y = df.cnt.values

    elif name == "adult":
        df = pd.read_csv("data/adult.csv").dropna()
        x = df.drop(
            columns=[
                "income",
                "occupation",
                "relationship",
            ]
        )
        categorical_variables = [
            "native-country",
            "workclass",
            "education",
            "marital-status",
            "race",
            "gender",
        ]

        for variable_name in categorical_variables:
            unique_values = df[variable_name].unique()
            for i, v in enumerate(unique_values):
                x.loc[x[variable_name] == v, variable_name] = i

        df.loc[df["income"] == ">50K", "income"] = 1
        df.loc[df["income"] == "<=50K", "income"] = 0
        Y = df["income"].astype("float32")
        sss = StratifiedShuffleSplit(n_splits=5, train_size=0.02, random_state=0)
        for train_index, test_index in sss.split(x, Y):
            print(train_index)
            X, _X_test = x.iloc[train_index, :].astype("float32"), None
            y, _y_test = Y[train_index], None
        CONSTANT = ["capital-loss", "capital-gain", "race", "gender"]

    else:
        raise NotImplementedError("Dataset name not found")

    return X, y, CONSTANT


if __name__ == "__main__":
    args = arguments()
    np.random.seed(args.seed)
    tf.get_logger().setLevel("ERROR")
    tf.random.set_seed(args.seed)

    X, y, CONSTANT = get_dataset(args.name)

    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(X)

    model = tf.keras.Sequential()
    model.add(normalizer)
    model.add(tf.keras.layers.Dense(args.size, activation="relu"))
    model.add(tf.keras.layers.Dense(args.size, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["acc", "AUC"],
    )
    model.fit(X, y, batch_size=32, epochs=50, verbose=1)

    explainer = code.Explainer(model, X, constrain=args.constrain)

    alg = code.GradientAlgorithm(
        explainer,
        variable=args.variable,
        constant=CONSTANT,
        learning_rate=args.lr,
        explanation_names=["pd", "ale", "pd_tf"],
        dist_weight=args.dist_weight,
    )

    if args.strategy == "target":
        alg.fool_aim(max_iter=args.iter, random_state=args.seed)
    else:
        alg.fool(max_iter=args.iter, random_state=args.seed)

    BASE_DIR = f"results_final/{args.name}/{args.variable}_{args.size}_{args.seed}_gradient_{args.lr}_{args.iter}_{args.dist_weight}"
    if args.constrain:
        BASE_DIR += "_constrained"
    os.makedirs(BASE_DIR, exist_ok=True)

    alg.plot_losses(savefig=f"{BASE_DIR}/loss")
    alg.plot_explanation(savefig=f"{BASE_DIR}/expl")
    alg.plot_data(constant=False, savefig=BASE_DIR)
    alg.get_metrics(args, f"{BASE_DIR}/")
