# ---------------------------------
# example: bike / NN / variable
# > python bike-sharing.py
# ---------------------------------

import os
import tensorflow as tf
tf.experimental.numpy.experimental_enable_numpy_behavior(
    prefer_float32=False
)
import argparse
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--variable', default="hr", type=str, help='variable')
parser.add_argument('--strategy', default="target", type=str, help='strategy type')
parser.add_argument('--iter', default=50, type=int, help='max iterations')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--method', default="ale", type=str, help='method: pd/ale')
parser.add_argument('--load_model', default=None, type=str, help='name of previously trained model')
parser.add_argument('--save_model_as', default=None, type=str, help='name of model to save')
parser.add_argument('--models_path', default='models', type=str, help='dirname of models')
parser.add_argument('--mse_regularization_factor', default=100, type=int, help='Regularization factor of MSE')
parser.add_argument('--alepp_loss_funcion', default="bce", type=str, help='loss function bce/mse')
args = parser.parse_args()

VARIABLE = args.variable
VARIABLES_TO_DROP = ['instant', 'dteday', 'season', 'casual', 'registered', 'cnt']


tf.get_logger().setLevel('ERROR')
tf.random.set_seed(args.seed)

import src
import numpy as np
np.random.seed(args.seed)
import pandas as pd

df = pd.read_csv("data/hour.csv")
X, y = df.drop(VARIABLES_TO_DROP, axis=1), df.cnt.values

if args.load_model:
    model = tf.keras.models.load_model( args.models_path + "/" + args.load_model)
else:
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(X)

    model = tf.keras.Sequential()
    model.add(normalizer)
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['mse'])
    model.fit(X, y, batch_size=256, epochs=50, verbose=1)

    if args.save_model_as:
        os.makedirs(args.models_path, exist_ok=True)
        model.save(args.models_path + "/" + args.save_model_as)

explainer = src.Explainer(model, X, y)

VARIABLES = {
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal'
}
CONSTANT = [
    'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday'
]

alg = src.GradientAlgorithm(
    explainer,
    variable=VARIABLE,
    constant=CONSTANT,
    learning_rate=0.5
)

if args.strategy == "target":
    alg.fool_aim(max_iter=args.iter, random_state=args.seed, method=args.method)
else:
    alg.fool(max_iter=args.iter, random_state=args.seed, method=args.method)


print( alg.explainer.model(X[:20]) )
alg.plot_losses()
title = "Partial Dependence" if args.method == "pd" else "Accumulated Local Effects"
alg.plot_explanation(method=args.method, title=title)
alg.plot_data(constant=False)

#----------------------- ale++ --------------------------

X_changed = pd.DataFrame(data=alg._X_changed, columns=X.columns)
explainer_alepp = src.Explainer(alg.explainer.model, X_changed, y)

alg_alepp = src.GradientAlgorithm(
    explainer_alepp,
    variable=VARIABLE,
    constant=CONSTANT,
    learning_rate=0.5
)

alg_alepp.fool_acc(max_iter=args.iter, reg_factor=args.mse_regularization_factor)
alg_alepp.plot_losses()
alg_alepp.plot_explanation(method='ale++', title="ale++")
