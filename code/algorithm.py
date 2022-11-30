import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Algorithm:
    def __init__(
        self,
        explainer,
        variable,
        constant=None,
        n_grid_points=21,
        explanation_names=["pd", "ale"],
    ):

        self.explainer = explainer
        self._variable = variable
        self._n_grid_points = n_grid_points

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
            self.result_explanations[name] = {
                "grid": None,
                "original": None,
                "changed": None,
            }
            self.iter_losses["loss"][name] = []
            self.iter_explanations[name] = {}

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
        else:
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
                X=self._X, idv=self._idv, grid=result_explanation["grid"]
            )

            result_explanation["changed"] = np.zeros_like(result_explanation["grid"])

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
                plt.plot(
                    result_explanation["grid"],
                    result_explanation["original"],
                    color="#000000",
                    lw=lw,
                )
                explanation_func = getattr(self.explainer, explanation_name)
                for i in range(n):
                    plt.plot(
                        result_explanation["grid"],
                        explanation_func(
                            self.get_best_data(i), self._idv, result_explanation["grid"]
                        ),
                        color=_colors[i],
                        lw=2,
                    )
                if target and self._aim:
                    plt.plot(
                        result_explanation["grid"],
                        result_explanation["target"],
                        color="#FF0000",
                        lw=lw,
                    )
                    leg = plt.legend(
                        [
                            "original",
                            *["changed-" + str(i) for i in range(n)],
                            "target",
                        ],
                        fontsize=14,
                        loc=legend_loc,
                    )
                else:
                    leg = plt.legend(
                        ["original", *["changed-" + str(i) for i in range(n)]],
                        fontsize=14,
                        loc=legend_loc,
                    )
            for i, _ in enumerate(leg.get_lines()):
                leg.get_lines()[i].set_linewidth(lw)
            plt.title(explanation_name.upper(), fontsize=20)
            plt.xlabel("variable: " + self._variable, fontsize=16)
            plt.ylabel("prediction", fontsize=16)
            if savefig:
                plt.savefig(savefig)
            plt.show()

    def plot_data(self, i=0, constant=True, height=2, savefig=None):
        plt.rcParams["legend.handlelength"] = 0.1
        _colors = sns.color_palette("Set1").as_hex()[0:2][::-1]
        if i == 0:
            _df = self.result_data
        else:
            _data_changed = pd.DataFrame(
                self.get_best_data(i), columns=self.explainer.data.columns
            )
            _df = (
                pd.concat((self.explainer.data, _data_changed))
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
        if savefig:
            ax.savefig(savefig, bbox_inches="tight")
        plt.show()

    def plot_losses(self, lw=3, figsize=(9, 6), savefig=None):
        for explanation_name in self.iter_losses["loss"].keys():
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
                plt.savefig(savefig)
            plt.show()
