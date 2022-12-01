import numpy as np
import pandas as pd
import warnings

class Explainer:
    def __init__(self, model, data, predict_function=None):
        self.model = model

        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, np.ndarray):
            warnings.warn("`data` is a numpy.ndarray -> coercing to pandas.DataFrame.")
            self.data = pd.DataFrame(data)
        else:
            raise TypeError(
                "`data` is a "
                + str(type(data))
                + ", and it should be a pandas.DataFrame."
            )

        if predict_function:
            self.predict_function = predict_function
        else:
            # scikit-learn extraction
            if hasattr(model, "_estimator_type"):
                if model._estimator_type == "classifier":
                    self.predict_function = lambda m, d: m.predict_proba(d)[:, 1]
                elif model._estimator_type == "regressor":
                    self.predict_function = lambda m, d: m.predict(d)
                else:
                    raise ValueError(
                        "Unknown estimator type: " + str(model._estimator_type) + "."
                    )
            # tensorflow extraction
            elif str(type(model)).startswith("<class 'tensorflow.python.keras.engine"):
                if model.output_shape[1] == 1:
                    self.predict_function = lambda m, d: m.predict(np.array(d)).reshape(
                        -1,
                    )
                elif model.output_shape[1] == 2:
                    self.predict_function = lambda m, d: m.predict(np.array(d))[:, 1]
                else:
                    warnings.warn(
                        "`model` predict output has shape greater than 2, predicting column 1."
                    )
            # default extraction
            else:
                if hasattr(model, "predict_proba"):
                    self.predict_function = lambda m, d: m.predict_proba(d)[:, 1]
                elif hasattr(model, "predict"):
                    self.predict_function = lambda m, d: m.predict(d)
                else:
                    raise ValueError(
                        "`predict_function` can't be extracted from the model. \n"
                        + "Pass `predict_function` to the Explainer, e.g. "
                        + "lambda m, d: m.predict(d), which returns a (1d) numpy.ndarray."
                    )

        try:
            pred = self.predict(data.values)
        except:
            raise ValueError("`predict_function(model, data)` returns an error.")
        if not isinstance(pred, np.ndarray):
            raise TypeError(
                "`predict_function(model, data)` returns an object of type "
                + str(type(pred))
                + ", and it must return a (1d) numpy.ndarray."
            )
        if len(pred.shape) != 1:
            raise ValueError(
                "`predict_function(model, data` returns an object of shape "
                + str(pred.shape)
                + ", and it must return a (1d) numpy.ndarray."
            )

    def predict(self, data):
        return self.predict_function(self.model, data)

    # ************* pd *************** #

    def pd(self, X, idv, grid):
        """
        numpy implementation of pd calculation for 1 variable

        takes:
        X - np.ndarray (2d), data
        idv - int, index of variable to calculate profile

        returns:
        y - np.ndarray (1d), vector of pd profile values
        """

        grid_points = len(grid)
        # take grid_points of each observation in X
        X_long = np.repeat(X, grid_points, axis=0)
        # take grid for each observation
        grid_long = np.tile(grid.reshape((-1, 1)), (X.shape[0], 1))
        # merge X and grid in long format
        X_long[:, [idv]] = grid_long
        # calculate ceteris paribus
        y_long = self.predict(X_long)
        # calculate partial dependence
        y = y_long.reshape(X.shape[0], grid_points).mean(axis=0)

        return y

    def pd_pop(self, X_pop, idv, grid):
        """
        vectorized (whole population) pd calculation for 1 variable
        """
        grid_points = len(grid)
        # take grid_points of each observation in X
        X_pop_long = np.repeat(X_pop, grid_points, axis=1)
        # take grid for each observation
        grid_pop_long = np.tile(
            grid.reshape((-1, 1)), (X_pop.shape[0], X_pop.shape[1], 1)
        )
        # merge X and grid in long format
        X_pop_long[:, :, [idv]] = grid_pop_long
        # calculate ceteris paribus
        y_pop_long = self.predict(
            X_pop_long.reshape(
                X_pop_long.shape[0] * X_pop_long.shape[1], X_pop_long.shape[2]
            )
        ).reshape((X_pop_long.shape[0], X_pop.shape[1], grid_points))
        # calculate partial dependence
        y = y_pop_long.mean(axis=1)

        return y

    def ale(self, X, idv, grid):
        """
        numpy implementation of pd calculation for 1 variable

        takes:
        X - np.ndarray (2d), data
        idv - int, index of variable to calculate profile

        returns:
        y - np.ndarray (1d), vector of pd profile values
        """

        import copy

        grid_copy = copy.deepcopy(grid)
        X_copy = copy.deepcopy(X)
        idv_copy = copy.deepcopy(idv)
        # print(grid)

        grid_points = len(grid)
        grid_copy[0]-=0.0001
        grid_copy[-1]+=0.0001
        # TODO przypadek większe niż ostatni punkt z grida (osobno?)
        # X: (liczba obserwacji, liczba featureów)

        #grid_pairs = zip(grid, grid[1:])



        bins = grid_copy

        X_copy_2 = copy.deepcopy(X_copy)

        b = np.digitize(X[:, idv_copy], bins, right=True)
        
        X_copy[:,idv_copy] = grid_copy[b-1]
        X_copy_2[:,idv_copy] = grid_copy[b]
        diff = self.predict(X_copy_2) - self.predict(X_copy)
        # print(diff[X_copy[:,idv]==grid[0]])
        # print(diff[X_copy[:,idv]==grid[0]])

        local_effects = np.nan_to_num(np.bincount(b-1, weights = diff, minlength=20) / np.bincount(b-1, minlength=20))
        y = np.cumsum(local_effects)
        z = copy.deepcopy(y)
        z = np.append(z, [z[-1]])

        # grid = grid_copy
        # # X_sorted = X[X[:, idv].argsort()]
        # local_effects = np.zeros_like(grid)

        # for i, (first, second) in enumerate(zip(grid, grid[1:])):
        #     # TODO edge case gdzie neighbors jest puste
        #     neighbors = X[(X[:, idv] > first) & (X[:, idv] <= second)]

        #     if neighbors.shape[0] == 0:
        #         continue

        #     neighbors_first = np.copy(neighbors)
        #     neighbors_second = np.copy(neighbors)
        #     neighbors_first[:, idv] = first
        #     neighbors_second[:, idv] = second
        #     # print(neighbors_first)
        #     # print(neighbors_second)
        #     # print(self.predict(neighbors_second) - self.predict(neighbors_first))
        #     #assert False
        #     result = (
        #         self.predict(neighbors_second) - self.predict(neighbors_first)
        #     ).mean()
        #     local_effects[i] = result

        # y = np.cumsum(local_effects)
        # print('y')
        # print(y)
        # print('z')
        # print(z)

        # assert np.allclose(y, z, atol = 1e-6), f'y={y}, z={z}, y-z={y-z}'

        # print(y-z)

        return z
