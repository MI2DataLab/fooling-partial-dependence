import numpy as np
import pandas as pd
import tqdm
from copy import deepcopy
import os

from . import algorithm
from . import loss
from . import utils

from scipy import stats

try:
    import tensorflow as tf
except:
    import warnings

    warnings.warn("`import tensorflow as tf` returns an error: gradient.py won't work.")

from code.explainer import sigmoid, logit


class GradientAlgorithm(algorithm.Algorithm):
    def __init__(
        self,
        explainer,
        variable,
        constant=None,
        n_grid_points=21,
        learning_rate=1e-2,
        explanation_names=["pd", "ale"],
        **kwargs,
    ):
        super().__init__(
            explainer=explainer,
            variable=variable,
            constant=constant,
            n_grid_points=n_grid_points,
            explanation_names=explanation_names,
        )

        params = dict(
            epsilon=1e-5,
            stop_iter=10,
            learning_rate=learning_rate,
            optimizer=utils.AdamOptimizer(),
            ks_weight = 0,
        )

        for k, v in kwargs.items():
            params[k] = v

        self.params = params

    def fool(
        self,
        grid=None,
        max_iter=50,
        random_state=None,
        save_iter=False,
        verbose=True,
        aim=False,
        center=None,
    ):

        self._aim = aim
        self._center = not aim if center is None else center
        self._result_data = {}


        # for j, (explanation_name, result_explanation) in enumerate(
        #     zip(self.result_explanations.keys(), self.result_explanations.values())
        # ):
        j = 0
        explanation_name = "pd_tf"
        result_explanation = self.result_explanations[explanation_name]

        if aim is False:
            super().fool(grid=grid, random_state=random_state)

        # init algorithm
        self._initialize()
        explanation_func = getattr(self.explainer, explanation_name)

        result_explanation["changed"] = explanation_func(
            self._X_changed, self._idv, result_explanation["grid"]
        )

        if j > 0:
            self.append_losses(explanation_name)
        else:
            self.append_losses(explanation_name, i=0)
        if save_iter:
            self.append_explanations(explanation_name, i=0)

        pbar = tqdm.tqdm(range(1, max_iter + 1), disable=not verbose)
        for i in pbar:
            # gradient of output w.r.t input
            result_explanation["changed"] = explanation_func(
                self._X_changed, self._idv, result_explanation["grid"]
            )
            loss_tf = self._calculate_gradient_tf(self._X_changed)
            step = self.params["optimizer"].calculate_step(loss_tf)
            self._X_changed -= self.params["learning_rate"] * step

            if j > 0:
                self.append_losses(explanation_name)
            else:
                self.append_losses(explanation_name, i=i)
            if save_iter:
                self.append_explanations(explanation_name, i=i)
            pbar.set_description(
                "Iter: %s || Loss: %s"
                % (i, self.iter_losses["loss"][explanation_name][-1])
            )
            if utils.check_early_stopping(
                self.iter_losses, self.params["epsilon"], self.params["stop_iter"]
            ):
                break

        result_explanation["changed"] = explanation_func(
            X=self._X_changed, idv=self._idv, grid=result_explanation["grid"]
        )

        # self.result_data[explanation_name] = self._X_changed
        _X_changed = deepcopy(self._X_changed)
        # if self.explainer.constrain:
        #     for i in range(_X_changed.shape[1]):
        #         _X_changed[:,i] = sigmoid(_X_changed[:,i])
        #         _X_changed[:,i] = self.explainer.unnormalizator[i](_X_changed[:,i])

        _data_changed = pd.DataFrame(
            _X_changed, columns=self.explainer.data.columns
        )

        self.result_explanations["ale"]["changed"] = self.explainer.ale(
            X=self._X_changed, idv=self._idv, grid=result_explanation["grid"]
        )

        # self.result_explanations["ale_dalex"]["changed"] = self.explainer.ale_dalex(
        #     X=self._X_changed, idv=self._idv, grid=result_explanation["grid"]
        # )

        self.result_explanations["pd"]["changed"] = self.explainer.pd(
            X=self._X_changed, idv=self._idv, grid=result_explanation["grid"]
        )

        

        self.result_data[explanation_name] = (
            pd.concat((self.explainer.original_data, _data_changed))
            .reset_index(drop=True)
            .rename(index={"0": "original", "1": "changed"})
            .assign(
                dataset=pd.Series(["original", "changed"])
                .repeat(self._n)
                .reset_index(drop=True)
            )
        )


    def fool_aim(
        self,
        target="auto",
        grid=None,
        max_iter=50,
        random_state=None,
        save_iter=False,
        verbose=True,
    ):
        super().fool_aim(target=target, grid=grid, random_state=random_state)
        self.fool(
            grid=None,
            max_iter=max_iter,
            random_state=random_state,
            save_iter=save_iter,
            verbose=verbose,
            aim=True,
        )

    #:# inside

    def _calculate_gradient_tf(self, data):
        # gradient of output w.r.t input
        input = tf.convert_to_tensor(data)

        with tf.GradientTape() as t:
            t.watch(input)
            explanation = self.explainer.pd_tf(X=input, idv=self._idv, grid=self.result_explanations["pd"]["grid"])
            loss_expl = loss.loss_tf(self.result_explanations["pd"]["target"], explanation, self._aim, self._center)
            loss_data = loss.loss_ks(self._X_original, input)
            print("loss_expl", loss_expl)
            print("loss_data", loss_data)
            loss_ = loss_expl + self.params["ks_weight"] * loss_data
            print("loss_", loss_)
            d_output_input = t.gradient(loss_, input).numpy()
            
        return d_output_input

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
        data_copy = deepcopy(data)
        if self.explainer.constrain:
            for i in range(data_copy.shape[1]):
                data_copy[:, i] = sigmoid(data_copy[:, i])
                data_copy[:, i] = self.explainer.unnormalizator[i](data_copy[:, i])

        data_long = np.repeat(data_copy, self._n_grid_points, axis=0)
        # take splits for each observation
        grid_long = np.tile(result_explanation["grid"], self._n)
        data_long[:, self._idv] = grid_long
        # merge X and splits in long format
        d_output_input_long = self._calculate_gradient(data_long)
        return d_output_input_long

    def _calculate_gradient_loss(self, result_explanation, d):
        # d = d_output_input_long
        d = d.reshape(self._n, self._n_grid_points, self._p)
        if self._aim:
            d_loss = (
                d
                * (
                    result_explanation["changed"].numpy() - result_explanation["target"].numpy()
                ).reshape(1, -1, 1)
            ).mean(axis=1)
        else:
            if self._center:
                d_loss = -(
                    (d - d.mean(axis=1).reshape(self._n, 1, self._p))
                    * (
                        (
                            result_explanation["changed"]
                            - result_explanation["changed"].mean()
                        )
                        - (
                            result_explanation["original"]
                            - result_explanation["original"].mean()
                        )
                    ).reshape(1, -1, 1)
                ).mean(axis=1)
            else:
                d_loss = -(
                    d
                    * (
                        result_explanation["changed"] - result_explanation["original"]
                    ).reshape(1, -1, 1)
                ).mean(axis=1)
        d_loss = d_loss / self._n
        d_loss[:, self._idv] = 0
        if self._idc is not None:
            d_loss[:, self._idc] = 0
        return d_loss

    def _initialize(self):
        _X_std = self._X.std(axis=0) * 1 / 9
        _X_std[self._idv] = 0
        if self._idc is not None:
            for c in self._idc:
                _X_std[c] = 0
        _theta = np.random.normal(loc=0, scale=_X_std, size=self._X.shape)
        self._X_changed = self._X + _theta

    #:# helper

    def append_losses(self, explanation_name, i=None):
        _loss = loss.loss_tf(
            original=self.result_explanations[explanation_name]["target"]
            if self._aim
            else self.result_explanations[explanation_name]["original"],
            changed=self.result_explanations[explanation_name]["changed"],
            aim=self._aim,
            center=self._center,
        )

        loss_data = loss.loss_ks(self._X_original, self._X_changed)
        _loss += self.params["ks_weight"] * loss_data

        if i is not None:
            self.iter_losses["iter"].append(i)
        self.iter_losses["loss"][explanation_name].append(_loss)

    def append_explanations(self, explanation_name, i=0):
        self.iter_explanations[explanation_name][i] = self.result_explanations[
            explanation_name
        ]["changed"]

    def get_metrics(self, args, save_path=None, all_results_csv="results/all_rows.csv"):
        output_str = ""
        df = pd.DataFrame(
            columns=[
                "name",
                "path",
                "variable",
                "size",
                "seed",
                "lr",
                "iter",
                "constrain",
                "ks_weight",
                "ale_l2",
                "ale_l1",
                "ale_max_diff",
                "ale_rho",
                "pd_l2",
                "pd_l1",
                "pd_max_diff",
                "pd_rho",
            ]
        )
        names = ("ALE", "PD")
        expls = [self.result_explanations["ale"], self.result_explanations["pd"]]

        new_row = [args.name, save_path, args.variable, args.size, args.seed, args.lr, args.iter, args.constrain, args.ks_weight]
        for name, explanation in zip(names, expls):

            l2 = np.sqrt((explanation["original"] - explanation["changed"])**2).mean()

            output_str += f"{name} L2: {l2}\n"

            l1 = np.abs(explanation["original"] - explanation["changed"]).mean()
            output_str += f"{name} L1: {l1}\n"

            max_diff = np.abs(explanation["original"] - explanation["changed"]).max()
            output_str += f"{name} max diff: {max_diff}\n"

            spearman_r, _ = stats.spearmanr(
                explanation["original"], explanation["changed"]
            )
            output_str += f"{name} Spearman rho: {spearman_r}\n"

            new_row.extend([l2, l1, max_diff, spearman_r])
        
        df.loc[len(df.index)] = new_row
        print(df)
        header = not os.path.isfile(all_results_csv) 
        df.to_csv(all_results_csv, mode='a', header=header)
        print(output_str)
        if save_path:
            with open(save_path + "metrics.txt", "w") as text_file:
                text_file.write(output_str)
