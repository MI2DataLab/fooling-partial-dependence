import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf


class Algorithm:
    def __init__(
        self,
        explainer,
        variable,
        constant=None,
        n_grid_points=21,
        explanation_names=["pd", "ale", "ale_dalex", "pd_tf"],
    ):

        self.explainer = explainer
        self._variable = variable
        self._n_grid_points = n_grid_points

        self._X_original = tf.convert_to_tensor(explainer.original_data.values)
        print("SHAPE", self._X_original.shape)
        self._X = explainer.data.values
        self._n, self._p = self._X.shape
        self._idv = explainer.data.columns.get_loc(variable)

        if constant is not None:
            self._idc = []
            for const in constant:
                self._idc.append(explainer.data.columns.get_loc(const))
        else:
            self._idc = None

        self.iter_losses = {"iter": [], "loss": {}}

        self.result_explanations, self.iter_explanations = {}, {}
        for name in explanation_names:
        # for name in ("pd", ):
            self.result_explanations[name] = {
                "grid": None,
                "original": None,
                "changed": None,
            }

# data shape: (nsamples, nfeatures)

        self.iter_losses["loss"]["pd"] = []
        self.iter_losses["loss"]["pd_tf"] = []

        self.iter_explanations["pd"] = {}

        self.result_data = {}

    def fool(self, grid=None, random_state=None):

        if random_state is not None:
            np.random.seed(random_state)

        if grid is None:
            for result_explanation in self.result_explanations.values():
                result_explanation["grid"] = np.linspace(
                    self._X[:, self._idv].min(),
                    self._X[:, self._idv].max(),
                    self._n_grid_points,
                )
                # if self.explainer.constrain:
                #     result_explanation["grid_original"] = np.linspace(
                #                         self._X_original[:, self._idv].min(),
                #                         self._X_original[:, self._idv].max(),
                #                         self._n_grid_points,
                #     )
        else:
            NotImplementedError()
            if not isinstance(grid, np.ndarray):
                raise TypeError("`grid` needs to be a np.ndarray")
            for result_explanation in self.result_explanations.values():
                result_explanation["grid"] = grid
            self._n_grid_points = len(grid)

        for explanation_name, result_explanation in zip(
            self.result_explanations.keys(), self.result_explanations.values()
        ):
            explanation_func = getattr(self.explainer, explanation_name)
            result_explanation["original"] = explanation_func(
                X=np.array(self._X),
                idv=self._idv,
                grid=result_explanation["grid"],
            )


            result_explanation["changed"] = np.zeros_like(result_explanation["grid"])

        # assert False

    def fool_aim(self, target="auto", grid=None, random_state=None):

        Algorithm.fool(self=self, grid=grid, random_state=random_state)

        if target == "auto":  # target = -(x - mean(x)) + mean(x)
            for result_explanation in self.result_explanations.values():
                result_explanation["target"] = np.mean(
                    result_explanation["original"]
                ) - (
                    result_explanation["original"]
                    - np.mean(result_explanation["original"])
                )
        elif isinstance(target, np.ndarray):
            for result_explanation in self.result_explanations.values():
                result_explanation["target"] = target
        else:  # target is a function
            for result_explanation in self.result_explanations.values():
                result_explanation["target"] = target(result_explanation["grid"])

    #:# plots

    def plot_explanation(
        self,
        target=True,
        n=1,
        lw=3,
        categorical=False,
        legend_loc=0,
        figsize=(9, 6),  # 7.2, 4.8
        savefig=None,
    ):
        plt.rcParams["legend.handlelength"] = 2
        plt.rcParams["figure.figsize"] = figsize
        _colors = sns.color_palette("Set2").as_hex()

        for explanation_name, result_explanation in zip(
            self.result_explanations.keys(), self.result_explanations.values()
        ):
            if n == 1:
                if categorical:

                    _df = pd.DataFrame(result_explanation)
                    _df = pd.melt(
                        _df,
                        id_vars=["grid"],
                        value_vars=["original", "changed"],
                        var_name="dataset",
                        value_name="prediction",
                    )
                    _df.grid = _df.grid.astype(int)
                    sns.barplot(
                        x="grid",
                        y="prediction",
                        hue="dataset",
                        data=_df,
                        palette=sns.color_palette("Set1").as_hex()[0:2][::-1],
                    )
                else:
                    _df = pd.DataFrame(result_explanation).set_index("grid")
                    if "target" not in _df.columns:
                        sns.lineplot(
                            data=_df,
                            linewidth=lw,
                            palette=sns.color_palette("Set1").as_hex()[0:2][::-1],
                        )
                    elif target is False:
                        sns.lineplot(
                            data=_df.drop("target", axis=1),
                            linewidth=lw,
                            palette=sns.color_palette("Set1").as_hex()[0:2][::-1],
                        )
                    else:
                        sns.lineplot(
                            data=_df,
                            linewidth=lw,
                            palette=sns.color_palette("Set1").as_hex()[0:2][::-1]
                            + ["grey"],
                        )
                leg = plt.legend(fontsize=14, loc=legend_loc)
            else:
                pass
                # the code that was here was calling for a function that doesn't exist anyway
                # TODO cleanup

            for i, _ in enumerate(leg.get_lines()):
                leg.get_lines()[i].set_linewidth(lw)
            plt.title(explanation_name.upper(), fontsize=20)
            plt.xlabel("variable: " + self._variable, fontsize=16)
            plt.ylabel("prediction", fontsize=16)
            if savefig:
                plt.savefig(f"{savefig}_{explanation_name}.png")
            # plt.show()
            plt.clf()

    def plot_other_explanation(
        self,
        target=True,
        n=1,
        lw=3,
        categorical=False,
        legend_loc=0,
        figsize=(9, 6),  # 7.2, 4.8
        savefig=None,
    ):
        plt.rcParams["legend.handlelength"] = 2
        plt.rcParams["figure.figsize"] = figsize
        _colors = sns.color_palette("Set2").as_hex()

        for explanation_name in self.result_explanations.keys():
            other_name = "ale" if explanation_name == "pd" else "pd"
            other_explanation_func = getattr(self.explainer, other_name)
            data = self.result_data[explanation_name]
            grid = self.result_explanations[explanation_name]["grid"]
            result_explanation = {"grid": grid}

            for key in ("original", "changed"):
                data_ = data[data.dataset == key].drop("dataset", axis=1).to_numpy()
                result_explanation[key] = other_explanation_func(
                    X=data_, idv=self._idv, grid=grid
                )

            result_explanation["target"] = self.result_explanations[other_name][
                "target"
            ]  # TODO is this right?

            if n == 1:
                if categorical:
                    _df = pd.DataFrame(result_explanation)
                    _df = pd.melt(
                        _df,
                        id_vars=["grid"],
                        value_vars=["original", "changed"],
                        var_name="dataset",
                        value_name="prediction",
                    )
                    _df.grid = _df.grid.astype(int)
                    sns.barplot(
                        x="grid",
                        y="prediction",
                        hue="dataset",
                        data=_df,
                        palette=sns.color_palette("Set1").as_hex()[0:2][::-1],
                    )
                else:
                    _df = pd.DataFrame(result_explanation).set_index("grid")
                    if "target" not in _df.columns:
                        sns.lineplot(
                            data=_df,
                            linewidth=lw,
                            palette=sns.color_palette("Set1").as_hex()[0:2][::-1],
                        )
                    elif target is False:
                        sns.lineplot(
                            data=_df.drop("target", axis=1),
                            linewidth=lw,
                            palette=sns.color_palette("Set1").as_hex()[0:2][::-1],
                        )
                    else:
                        sns.lineplot(
                            data=_df,
                            linewidth=lw,
                            palette=sns.color_palette("Set1").as_hex()[0:2][::-1]
                            + ["grey"],
                        )
                leg = plt.legend(fontsize=14, loc=legend_loc)

            else:
                pass
                # the code that was here was calling for a function that doesn't exist anyway
                # TODO cleanup

            for i, _ in enumerate(leg.get_lines()):
                leg.get_lines()[i].set_linewidth(lw)
            plt.title(
                f"{other_name.upper()} optmised on {explanation_name.upper()}",
                fontsize=20,
            )
            plt.xlabel("variable: " + self._variable, fontsize=16)
            plt.ylabel("prediction", fontsize=16)
            if savefig:
                plt.savefig(f"{savefig}_{explanation_name}_other.png")
            plt.show()
            plt.clf()

    def plot_data(self, i=0, constant=True, height=2, savefig=None):
        # for explanation_name in self.result_explanations.keys():
        for explanation_name in ("pd_tf", ):
            plt.rcParams["legend.handlelength"] = 0.1
            _colors = sns.color_palette("Set1").as_hex()[0:2][::-1]
            if i == 0:
                _df = self.result_data[explanation_name]
            else:
                _data_changed = pd.DataFrame(
                    self.get_best_data(i), columns=self.explainer.data.columns
                )
                _df = (
                    pd.concat((self.explainer.original_data, _data_changed))
                    .reset_index(drop=True)
                    .rename(index={"0": "original", "1": "changed"})
                    .assign(
                        dataset=pd.Series(["original", "changed"])
                        .repeat(self._n)
                        .reset_index(drop=True)
                    )
                )
            if not constant and self._idc is not None:
                _df = _df.drop(_df.columns[self._idc], axis=1)
            ax = sns.pairplot(_df, hue="dataset", height=height, palette=_colors)
            ax._legend.set_bbox_to_anchor((0.62, 0.64))
            plt.title("Data - " + explanation_name.upper(), fontsize=20, loc="center")
            if savefig:
                ax.savefig(
                    os.path.join(savefig, f"data_{explanation_name}.png"),
                    bbox_inches="tight",
                )
            # plt.show()
            plt.clf()

    def plot_losses(self, lw=3, figsize=(9, 6), savefig=None):
        # for explanation_name in self.iter_losses["loss"].keys():
        for explanation_name in ("pd_tf", ):
            plt.rcParams["figure.figsize"] = figsize
            plt.plot(
                self.iter_losses["iter"],
                self.iter_losses["loss"][explanation_name],
                color="#000000",
                lw=lw,
            )
            plt.title("Learning curve - " + explanation_name.upper(), fontsize=20)
            plt.xlabel("epoch", fontsize=16)
            plt.ylabel("loss", fontsize=16)
            if savefig:
                plt.savefig(f"{savefig}_{explanation_name}.png")
            # plt.show()
            plt.clf()
