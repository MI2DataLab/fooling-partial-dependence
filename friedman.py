# ---------------------------------
# test the algorithm implementation
# on a multidimensional XOR problem
# > python xor.py
# ---------------------------------

import os
import tensorflow as tf
import argparse
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--algorithm', default="gradient", type=str, help='algorithm type')
parser.add_argument('--strategy', default="target", type=str, help='strategy type')
parser.add_argument('--center', default=None, type=str, help='center in strategy r')
parser.add_argument('--n', default=320, type=int, help='number of observations')
parser.add_argument('--dim', default=3, type=int, help='number of dimensions of X')
parser.add_argument('--size', default=16, type=int, help='number of neurons in layers')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate for gradient algorithm')
args = parser.parse_args()

N = args.n
SIZE = args.size
DIM = args.dim
SEED = args.seed
LR = args.lr
tf.random.set_seed(SEED)

import code
import numpy as np
import pandas as pd

# Friedman dataset as in https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html
np.random.seed(args.seed)
X = np.random.uniform(size=(N, 5))
X[:, DIM:] = 0.5
y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
y = (y > (y.mean())).astype(int)
X = X[:, :DIM]
X = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(DIM)])

normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
normalizer.adapt(X)

model = tf.keras.Sequential()
model.add(normalizer)
model.add(tf.keras.layers.Dense(SIZE, activation="relu"))
model.add(tf.keras.layers.Dense(SIZE, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['acc', 'AUC'])
model.fit(X, y, batch_size=int(N/10), epochs=300, verbose=0)

explainer = code.Explainer(model, X)

if args.algorithm == "gradient":
    alg = code.GradientAlgorithm(explainer, variable="x1", learning_rate=LR)
else:
    alg = code.GeneticAlgorithm(explainer, variable="x1", std_ratio=1/6)

if args.strategy == "target":
    alg.fool_aim(random_state=args.seed)
else:
    alg.fool(center=args.center, random_state=args.seed)

BASE_DIR = f"imgs/friedman/{DIM}_{SIZE}_{N}_{SEED}_{args.algorithm}_{LR}"
os.makedirs(BASE_DIR, exist_ok=True)

alg.plot_losses(savefig=f"{BASE_DIR}/loss")
alg.plot_explanation(savefig=f"{BASE_DIR}/expl")
alg.plot_data(constant=False, savefig=f"{BASE_DIR}/data.png")
alg.get_metrics(f"{BASE_DIR}/metrics.txt")