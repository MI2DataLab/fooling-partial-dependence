import numpy as np
import pandas as pd
import tqdm

from . import algorithm
from . import loss
from . import utils

try:
    import tensorflow as tf
except:
    import warnings
    warnings.warn("`import tensorflow as tf` returns an error: gradient.py won't work.")

class GradientAlgorithm(algorithm.Algorithm):
    def __init__(
        self,
        explainer,
        variable,
        constant=None,
        n_grid_points=21,
        loss_function='mse',
        X_poisoned=None,
        **kwargs
    ):
        super().__init__(
            explainer=explainer,
            variable=variable,
            constant=constant,
            n_grid_points=n_grid_points
        )

        params = dict(
            epsilon=1e-5,
            stop_iter=10,
            learning_rate=1e-2,
            optimizer=utils.AdamOptimizer()
        )

        for k, v in kwargs.items():
            params[k] = v

        self.params = params
        if loss_function == 'mse':
            self.loss_function = tf.keras.losses.MeanSquaredError()
        elif loss_function == 'bce':
            self.loss_function = tf.keras.losses.BinaryCrossentropy()
        else:
            raise ValueError("Loss function not recognized.")
        self.X_poisoned = X_poisoned


    def fool(
        self,
        grid=None,
        max_iter=50,
        random_state=None,
        save_iter=False,
        verbose=True,
        aim=False,
        center=None,
        method="pd",
    ):
        self._aim = aim
        self._center = not aim if center is None else center
        if aim is False:
            super().fool(grid=grid, random_state=random_state, method=method)

        # init algorithm
        self._initialize()
        if method == "pd":
            self.result_explanation['changed'] = self.explainer.pd(
                self._X_changed,
                self._idv,
                self.result_explanation['grid']
            )
        elif method == "ale":
            self.result_explanation['changed'] = self.explainer.ale(
                self._X_changed,
                self._idv,
                self.result_explanation['grid']
            )
        self.append_losses(i=0)
        if save_iter:
            self.append_explanations(i=0)

        pbar = tqdm.tqdm(range(1, max_iter + 1), disable=not verbose)
        for i in pbar:
            if method == "pd":
                self.result_explanation['changed'] = self.explainer.pd(
                    self._X_changed,
                    self._idv,
                    self.result_explanation['grid']
                )
            elif method == "ale":
                self.result_explanation['changed'] = self.explainer.ale(
                    self._X_changed,
                    self._idv,
                    self.result_explanation['grid']
                )

            gradient = self.calculate_gradient(self._X_changed, method=method)
            step = self.params['optimizer'].calculate_step(gradient)
            self._X_changed -= self.params['learning_rate'] * step

            self.append_losses(i=i)
            if save_iter:
                self.append_explanations(i=i)
            pbar.set_description("Iter: %s || Loss: %s" % (i, self.iter_losses['loss'][-1]))
            if utils.check_early_stopping(self.iter_losses, self.params['epsilon'], self.params['stop_iter']):
                break

        if method == "pd":
            self.result_explanation['changed'] = self.explainer.pd(
                X=self._X_changed,
                idv=self._idv,
                grid=self.result_explanation['grid']
            )
        elif method == "ale":
            self.result_explanation['changed'] = self.explainer.ale(
                X=self._X_changed,
                idv=self._idv,
                grid=self.result_explanation['grid']
            )
        _data_changed = pd.DataFrame(self._X_changed, columns=self.explainer.data.columns)
        self.result_data = pd.concat((self.explainer.data, _data_changed))\
            .reset_index(drop=True)\
            .rename(index={'0': 'original', '1': 'changed'})\
            .assign(dataset=pd.Series(['original', 'changed'])\
                            .repeat(self._n).reset_index(drop=True))
        self.X_poisoned = self._X_changed


    def fool_acc(self, max_iter=None, verbose=True, grid=None,random_state=None, method="ale++", reg_factor=100):
        """
        Function finetuning model weights for better accuracy, but preserving current ALE plot
        """

        super().fool(
            grid=grid,
            random_state=random_state,
            method="ale++",
        )

        target = tf.convert_to_tensor(self.result_explanation['original'])
        y = tf.convert_to_tensor(self._y, dtype=tf.float32)
        poisoned_data = tf.convert_to_tensor(self.X_poisoned)
        raw_loss, regularized_loss = self.calculate_alepp_loss(self._X, poisoned_data, y, reg_factor)
        self.iter_losses['iter'] = [0]
        self.iter_losses['raw_loss'].append(raw_loss)
        self.iter_losses['regularized_loss'].append(regularized_loss)

        self.explainer.model.evaluate(self._X, self._y)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate'])

        pbar = tqdm.tqdm(range(1, max_iter + 1), disable=not verbose)
        for i in pbar:
            data = tf.convert_to_tensor(self._X)
            poisoned_data = tf.convert_to_tensor(self.X_poisoned)
            labels = tf.convert_to_tensor(self._y, dtype=tf.float32)

            with tf.GradientTape() as t:
                t.watch(data)
                t.watch(target)
                t.watch(labels)

                raw_loss, regularized_loss = self.calculate_alepp_loss(data, poisoned_data, labels, reg_factor)
                gradient = t.gradient(regularized_loss, self.explainer.model.trainable_weights)

                if isinstance(gradient, tf.IndexedSlices):
                    gradient = tf.convert_to_tensor(gradient)

                optimizer.apply_gradients(zip(gradient, self.explainer.model.trainable_weights))
                self.iter_losses['iter'].append(i)
                self.iter_losses['raw_loss'].append(raw_loss)
                self.iter_losses['regularized_loss'].append(regularized_loss)

            pbar.set_description(f"Iter: {i} || Loss: {self.iter_losses['regularized_loss'][-1]} || Raw loss: {self.iter_losses['raw_loss'][-1]}")
            if utils.check_early_stopping(self.iter_losses, self.params['epsilon'], self.params['stop_iter']):
                break

        self.result_explanation['original_after_fintetuning'] = self.explainer.ale(
            self._X,
            self._idv,
            self.result_explanation['grid']
        )
        self.result_explanation['poisoned_after_finetuning'] = self.explainer.ale(
            self.X_poisoned,
            self._idv,
            self.result_explanation['grid']
        )
        self.explainer.model.evaluate(self._X, self._y)

    def fool_aim(
        self,
        target="auto",
        grid=None,
        max_iter=50,
        random_state=None,
        save_iter=False,
        verbose=True,
        method="pd",
    ):
        super().fool_aim(
            target=target,
            grid=grid,
            random_state=random_state,
            method=method,
        )
        self.fool(
            grid=None,
            max_iter=max_iter,
            random_state=random_state,
            save_iter=save_iter,
            verbose=verbose,
            aim=True,
            method=method,
        )


    #:# inside

    def calculate_pdp(self, data_tensor):
        data_long = tf.repeat(data_tensor, self._n_grid_points, axis=0)
        grid_long = tf.tile(tf.convert_to_tensor(self.result_explanation['grid']), tf.convert_to_tensor([self._n]))
        data_long = GradientAlgorithm.assign(data_long, (slice(None, None), self._idv), grid_long.reshape(-1, 1))
        return tf.reshape(self.explainer.model(data_long), (self._n, self._n_grid_points)).mean(axis=0)

    def calculate_ale(self, data_tensor):
        self.result_explanation['grid'][0] -= 0.000001
        self.result_explanation['grid'][-1] += 0.000001
        data_sorted_ids = tf.argsort(data_tensor[:, self._idv])
        data_sorted = tf.gather(data_tensor, data_sorted_ids, axis=0)

        z_idx = tf.searchsorted(data_sorted[:, self._idv], self.result_explanation['grid'])
        N = z_idx[1:] - z_idx[:-1]

        grid_points = len(self.result_explanation['grid'])

        lower = tf.repeat(self.result_explanation['grid'][:-1], N)
        upper = tf.repeat(self.result_explanation['grid'][1:], N)

        lower = tf.expand_dims(lower, axis=1)
        upper = tf.expand_dims(upper, axis=1)

        lower_data = GradientAlgorithm.assign(
            data_sorted,
            (slice(None, None), self._idv),
            lower
        )

        upper_data = GradientAlgorithm.assign(
            data_sorted,
            (slice(None, None), self._idv),
            upper
        )
        lower_pred = self.explainer.model(lower_data)
        upper_pred = self.explainer.model(upper_data)

        diff = upper_pred - lower_pred
        y = tf.zeros(grid_points)
        for k in range(1, grid_points):
            if N[k - 1] == 0:
                continue
            segment_average = tf.math.reduce_mean(diff[z_idx[k - 1]: z_idx[k]])

            y = GradientAlgorithm.assign(y, (k), segment_average)

        y = tf.math.cumsum(y)
        # N_with0 = tf.concat( [tf.zeros(1, dtype=tf.float32), tf.cast(N, tf.float32)], axis=0)
        # tf.tensordot(y, N_with0,1) / data_sorted.shape[0]
        # return y - tf.tensordot(y, N_with0,1) / data_sorted.shape[0]
        return y


    def calculate_loss(self, result):
        if self._aim:
            return tf.keras.losses.mean_squared_error(self.result_explanation['target'], result)
        else:
            assert False, "Not implemented"

    def calculate_gradient(self, data, method="pd"):
        input = tf.convert_to_tensor(data)
        with tf.GradientTape() as t:
            t.watch(input)
            if method == "pd":
                explanation = self.calculate_pdp(input)
            elif method == "ale":
                explanation = self.calculate_ale(input)
            loss = self.calculate_loss(explanation)
            gradient = t.gradient(loss, input)
            if isinstance(gradient, tf.IndexedSlices):
                gradient = tf.convert_to_tensor(gradient)

        return gradient.numpy()

    def assign(tensor, slc, values):
        '''
            Tensorflow can't do
                tensor[slc] = values
            this is a workaround
        '''

        mask = np.zeros_like(tensor.numpy())
        mask[slc] = 1
        mask = tf.convert_to_tensor(mask)
        return (1 - mask) * tensor + mask * values

    def assign2(tensor, slc, values):
        '''
            Tensorflow can't do
                tensor[slc] = values
            this is a workaround
        '''
        var = tf.Variable(tensor, trainable=True)
        with tf.control_dependencies([var[slc].assign(values)]):
                new_tensor = tf.identity(var)
        return new_tensor

    def _initialize(self):
        _X_std = self._X.std(axis=0) * 1/9
        _X_std[self._idv] = 0
        if self._idc is not None:
            for c in self._idc:
                _X_std[c] = 0
        _theta = np.random.normal(loc=0, scale=_X_std, size=self._X.shape)
        self._X_changed = self._X + _theta

    #:# helper

    def append_losses(self, i=0):
        _loss = loss.loss(
            original=self.result_explanation['target'] if self._aim else self.result_explanation['original'],
            changed=self.result_explanation['changed'],
            aim=self._aim,
            center=self._center
        )
        self.iter_losses['iter'].append(i)
        self.iter_losses['loss'].append(_loss)

    def append_explanations(self, i=0):
        self.iter_explanations[i] = self.result_explanation['changed']

    def calculate_alepp_loss(self, data, poisoned_data, y_orig, reg_factor):
        y_pred = self.explainer.model(data)
        alepp_loss = self.loss_function(y_pred, y_orig)
        explanation = self.calculate_ale(data)
        poisoned_explanation = self.calculate_ale(poisoned_data)
        reg_loss = tf.keras.losses.mean_squared_error(explanation, poisoned_explanation)
        return alepp_loss, alepp_loss + reg_factor*reg_loss
