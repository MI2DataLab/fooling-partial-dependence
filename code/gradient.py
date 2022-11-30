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
        explanation_names=['pd', 'ale'],
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

        self.result_explanations = {}
        for name in explanation_names:
            self.result_explanations[name] = {'grid': None, 'original': None, 'changed': None}


    def fool(
        self,
        grid=None,
        max_iter=50,
        random_state=None,
        save_iter=False,
        verbose=True,
        aim=False,
        center=None
    ):

        self._aim = aim
        self._center = not aim if center is None else center
        if aim is False:
            super().fool(grid=grid, random_state=random_state)

        # init algorithm
        self._initialize()

        for explanation_name, result_explanation in zip(self.result_explanations.keys(), self.result_explanations.values()):
            explanation_func = getattr(self.explainer, explanation_name)

            result_explanation['changed'] = explanation_func(
                self._X_changed, 
                self._idv, 
                result_explanation['grid']
            )
            self.append_losses(result_explanation, i=0)
            if save_iter:
                self.append_explanations(result_explanation, i=0)

            pbar = tqdm.tqdm(range(1, max_iter + 1), disable=not verbose)
            for i in pbar:
                # gradient of output w.r.t input
                _ = self._calculate_gradient(self._X_changed)
                d_output_input_long = self._calculate_gradient_long(result_explanation, self._X_changed)
                result_explanation['changed'] = explanation_func(
                    self._X_changed, 
                    self._idv, 
                    result_explanation['grid']
                )
                d_loss = self._calculate_gradient_loss(result_explanation, d_output_input_long)
                step = self.params['optimizer'].calculate_step(d_loss)
                self._X_changed -= self.params['learning_rate'] * step
                
                self.append_losses(result_explanation, i=i)
                if save_iter:
                    self.append_explanations(result_explanation, i=i)
                pbar.set_description("Iter: %s || Loss: %s" % (i, self.iter_losses['loss'][-1]))
                if utils.check_early_stopping(self.iter_losses, self.params['epsilon'], self.params['stop_iter']):
                    break

            result_explanation['changed'] = explanation_func(
                X=self._X_changed,
                idv=self._idv,
                grid=result_explanation['grid']
            )

        _data_changed = pd.DataFrame(self._X_changed, columns=self.explainer.data.columns)
        self.result_data = pd.concat((self.explainer.data, _data_changed))\
            .reset_index(drop=True)\
            .rename(index={'0': 'original', '1': 'changed'})\
            .assign(dataset=pd.Series(['original', 'changed'])\
                            .repeat(self._n).reset_index(drop=True))


    def fool_aim(
        self,
        target="auto",
        grid=None,
        max_iter=50,
        random_state=None,
        save_iter=False,
        verbose=True
    ):
        super().fool_aim(
            target=target,
            grid=grid,
            random_state=random_state
        )
        self.fool(
            grid=None,
            max_iter=max_iter, 
            random_state=random_state, 
            save_iter=save_iter, 
            verbose=verbose, 
            aim=True
        )


    #:# inside

    def _calculate_gradient(self, data):
        # gradient of output w.r.t input
        input = tf.convert_to_tensor(data)
        with tf.GradientTape() as t:
            t.watch(input)
            output = self.explainer.model(input)
            d_output_input = t.gradient(output, input).numpy()
        return d_output_input

    def _calculate_gradient_long(self, result_explanation, data):
        # gradient of output w.r.t input with changed idv to splits
        data_long = np.repeat(data, self._n_grid_points, axis=0)
        # take splits for each observation
        grid_long = np.tile(result_explanation['grid'], self._n)
        data_long[:, self._idv] = grid_long
        # merge X and splits in long format
        d_output_input_long = self._calculate_gradient(data_long)
        return d_output_input_long

    def _calculate_gradient_loss(self, result_explanation, d):
        # d = d_output_input_long
        d = d.reshape(self._n, self._n_grid_points, self._p)
        if self._aim:
            d_loss = (d * (result_explanation['changed'] - result_explanation['target']).reshape(1, -1, 1)).mean(axis=1) 
        else:
            if self._center:
                d_loss = - ((d - d.mean(axis=1).reshape(self._n, 1, self._p)) * \
                    ((result_explanation['changed'] - result_explanation['changed'].mean()) -\
                    (result_explanation['original'] - result_explanation['original'].mean())).reshape(1, -1, 1)).mean(axis=1)
            else:
                d_loss = - (d * (result_explanation['changed'] - result_explanation['original']).reshape(1, -1, 1)).mean(axis=1)  
        d_loss = d_loss / self._n
        d_loss[:, self._idv] = 0
        if self._idc is not None:
            d_loss[:, self._idc] = 0
        return d_loss

    def _initialize(self):
        _X_std = self._X.std(axis=0) * 1/9
        _X_std[self._idv] = 0
        if self._idc is not None:
            for c in self._idc:
                _X_std[c] = 0        
        _theta = np.random.normal(loc=0, scale=_X_std, size=self._X.shape)
        self._X_changed = self._X + _theta

    #:# helper
              
    def append_losses(self, result_explanation, i=0):
        _loss = loss.loss(
            original=result_explanation['target'] if self._aim else result_explanation['original'],
            changed=result_explanation['changed'],
            aim=self._aim,
            center=self._center
        )
        self.iter_losses['iter'].append(i)
        self.iter_losses['loss'].append(_loss)

    def append_explanations(self, result_explanation, i=0):
        self.iter_explanations[i] = result_explanation['changed']