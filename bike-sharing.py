# ---------------------------------
# example: heart / NN / variable
# > python heart-gradient.py
# ---------------------------------


import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description="main")
parser.add_argument("--variable", default="windspeed", type=str, help="variable")
parser.add_argument("--strategy", default="target", type=str, help="strategy type")
parser.add_argument("--iter", default=50, type=int, help="max iterations")
parser.add_argument("--seed", default=0, type=int, help="random seed")
args = parser.parse_args()
VARIABLE = args.variable

tf.get_logger().setLevel("ERROR")
tf.random.set_seed(args.seed)

import code
import numpy as np

np.random.seed(args.seed)
import pandas as pd

df = pd.read_csv("data/bike-sharing-day.csv")
variables_to_drop = [
    "instant",
    "dteday",
    "season",
    "weekday",
    "mnth",
]
df = df.drop(variables_to_drop, axis=1)

target_fields = ["cnt", "registered", "casual"]
X = df.drop(target_fields, axis=1)
y = df.cnt.values

normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
normalizer.adapt(X)

model = tf.keras.Sequential()
model.add(normalizer)
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["acc", "AUC"],
)
model.fit(X, y, batch_size=32, epochs=50, verbose=1)
explainer = code.Explainer(model, X)

VARIABLES = {"temp", "atemp", "humidity", "windspeed", "casual", "registered", "cnt"}
CONSTANT = {"yr", "workingday", "weathersit"}

alg = code.GradientAlgorithm(
    explainer, variable=VARIABLE, constant=CONSTANT, learning_rate=0.1
)

if args.strategy == "target":
    alg.fool_aim(max_iter=args.iter, random_state=args.seed)
else:
    alg.fool(max_iter=args.iter, random_state=args.seed)

alg.plot_losses()
alg.plot_explanation()
alg.plot_data(constant=False)
