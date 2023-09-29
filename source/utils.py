import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def dibuja_covar(data, titulo = 'Matriz de Covarianzas'):
    """
    Dibuja la matriz de covarianzas de un conjunto de datos
    :param data: Matriz de covarianzas
    :param titulo: Título del gráfico
    :return: 0
    """
    
        # Crear un gráfico de matriz de covarianzas
    plt.figure(figsize=(8, 6))

    vmin = 0
    vmax = 1
    plt.imshow(data, cmap='coolwarm', interpolation='nearest', vmin=vmin, vmax=vmax)

    # Mostrar los valores de covarianza en cada celda
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f'{data[i, j]:.2f}', va='center', ha='center', color='white')

    # Agregar límites de celdas con líneas negras
    for i in range(data.shape[0] + 1):
        plt.axhline(i - 0.5, color='black', linewidth=1)
        plt.axvline(i - 0.5, color='black', linewidth=1)

        # Agregar un cuadrado exterior
    outer_rect = patches.Rectangle((-0.5, -0.5), data.shape[0], data.shape[1], linewidth=2, edgecolor='black', facecolor='none')
    plt.gca().add_patch(outer_rect)

    # Añadir una barra de colores
    plt.colorbar(label='Covariance')
    plt.title(titulo)
    plt.xticks(np.arange(data.shape[0])), np.arange(1, data.shape[0] + 1)
    plt.yticks(np.arange(data.shape[1])), np.arange(1, data.shape[1] + 1)

    # Desactivar las líneas horizontales y verticales
    plt.grid(False)

    plt.show()

    return 0


def dibuja_covar_ax(data, ax):
    """
    Dibuja la matriz de covarianzas de un conjunto de datos en una figura con subgráficos
    :param data: Matriz de covarianzas
    :param ax: Subgráfico donde se dibujará la matriz de covarianzas
    :return: 0
    """ 
    # Crear un gráfico de matriz de covarianzas en el subgráfico especificado por 'ax'
    vmin = 0
    vmax = 1
    im = ax.imshow(data, cmap='coolwarm', interpolation='nearest', vmin=vmin, vmax=vmax)

    # Mostrar los valores de covarianza en cada celda
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f'{data[i, j]:.2f}', va='center', ha='center', color='white')

    # Agregar límites de celdas con líneas negras
    for i in range(data.shape[0] + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
        ax.axvline(i - 0.5, color='black', linewidth=1)

    # Agregar un cuadrado exterior
    outer_rect = patches.Rectangle((-0.5, -0.5), data.shape[0], data.shape[1], linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(outer_rect)

    # Añadir una barra de colores
    plt.colorbar(im, ax=ax, label='Covariance')
    ax.set_title('Matriz de Covarianzas')
    plt.xticks(np.arange(data.shape[0])), np.arange(1, data.shape[0] + 1)
    plt.yticks(np.arange(data.shape[1])), np.arange(1, data.shape[1] + 1)  


    # Desactivar las líneas horizontales y verticales
    ax.grid(False)

    return 0


""""""  #
"""
Copyright (c) 2020-2023, Dany Cajas
All rights reserved.
This work is licensed under BSD 3-Clause "New" or "Revised" License.
License available at https://github.com/dcajasn/Riskfolio-Lib/blob/master/LICENSE.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import scipy.stats as st
import scipy.cluster.hierarchy as hr
from scipy.spatial.distance import squareform
import networkx as nx
import riskfolio.src.RiskFunctions as rk
import riskfolio.src.AuxFunctions as af
import riskfolio.src.DBHT as db
import riskfolio.src.GerberStatistic as gs


__all__ = [
    "plot_series",
    "plot_frontier",
    "plot_pie",
    "plot_bar",
    "plot_frontier_area",
    "plot_risk_con",
    "plot_hist",
    "plot_range",
    "plot_drawdown",
    "plot_table",
    "plot_clusters",
    "plot_dendrogram",
    "plot_network",
]

rm_names = [
    "Standard Deviation",
    "Square Root Kurtosis",
    "Mean Absolute Deviation",
    "Gini Mean Difference",
    "Semi Standard Deviation",
    "Square Root Semi Kurtosis",
    "First Lower Partial Moment",
    "Second Lower Partial Moment",
    "Value at Risk",
    "Conditional Value at Risk",
    "Tail Gini",
    "Entropic Value at Risk",
    "Relativistic Value at Risk",
    "Worst Realization",
    "Conditional Value at Risk Range",
    "Tail Gini Range",
    "Range",
    "Max Drawdown",
    "Average Drawdown",
    "Drawdown at Risk",
    "Conditional Drawdown at Risk",
    "Entropic Drawdown at Risk",
    "Relativistic Drawdown at Risk",
    "Ulcer Index",
]

rmeasures = [
    "MV",
    "KT",
    "MAD",
    "GMD",
    "MSV",
    "SKT",
    "FLPM",
    "SLPM",
    "VaR",
    "CVaR",
    "TG",
    "EVaR",
    "RLVaR",
    "WR",
    "CVRG",
    "TGRG",
    "RG",
    "MDD",
    "ADD",
    "DaR",
    "CDaR",
    "EDaR",
    "RLDaR",
    "UCI",
]

def plot_frontier_modif(
    w_frontier,
    mu,
    cov=None,
    returns=None,
    rm="MV",
    kelly=False,
    rf=0,
    alpha=0.05,
    a_sim=100,
    beta=None,
    b_sim=None,
    kappa=0.30,
    solver=None,
    cmap="viridis",
    w=None,
    label="Portfolio",
    marker="*",
    s=16,
    c="r",
    height=6,
    width=10,
    t_factor=252,
    ax=None,
):
    r"""
    Creates a plot of the efficient frontier for a risk measure specified by
    the user.

    Parameters
    ----------
    w_frontier : DataFrame
        Portfolio weights of some points in the efficient frontier.
    mu : DataFrame of shape (1, n_assets)
        Vector of expected returns, where n_assets is the number of assets.
    cov : DataFrame of shape (n_features, n_features)
        Covariance matrix, where n_features is the number of features.
    returns : DataFrame of shape (n_samples, n_features)
        Features matrix, where n_samples is the number of samples and
        n_features is the number of features.
    rm : str, optional
        The risk measure used to estimate the frontier.
        The default is 'MV'. Possible values are:

        - 'MV': Standard Deviation.
        - 'KT': Square Root Kurtosis.
        - 'MAD': Mean Absolute Deviation.
        - 'MSV': Semi Standard Deviation.
        - 'SKT': Square Root Semi Kurtosis.
        - 'FLPM': First Lower Partial Moment (Omega Ratio).
        - 'SLPM': Second Lower Partial Moment (Sortino Ratio).
        - 'CVaR': Conditional Value at Risk.
        - 'TG': Tail Gini.
        - 'EVaR': Entropic Value at Risk.
        - 'RLVaR': Relativistic Value at Risk.
        - 'WR': Worst Realization (Minimax).
        - 'CVRG': CVaR range of returns.
        - 'TGRG': Tail Gini range of returns.
        - 'RG': Range of returns.
        - 'MDD': Maximum Drawdown of uncompounded returns (Calmar Ratio).
        - 'ADD': Average Drawdown of uncompounded cumulative returns.
        - 'DaR': Drawdown at Risk of uncompounded cumulative returns.
        - 'CDaR': Conditional Drawdown at Risk of uncompounded cumulative returns.
        - 'EDaR': Entropic Drawdown at Risk of uncompounded cumulative returns.
        - 'RLDaR': Relativistic Drawdown at Risk of uncompounded cumulative returns.
        - 'UCI': Ulcer Index of uncompounded cumulative returns.

    kelly : bool, optional
        Method used to calculate mean return. Possible values are False for
        arithmetic mean return and True for mean logarithmic return. The default
        is False.
    rf : float, optional
        Risk free rate or minimum acceptable return. The default is 0.
    alpha : float, optional
        Significance level of VaR, CVaR, EVaR, RLVaR, DaR, CDaR, EDaR, RLDaR and Tail Gini of losses.
        The default is 0.05.
    a_sim : float, optional
        Number of CVaRs used to approximate Tail Gini of losses. The default is 100.
    beta : float, optional
        Significance level of CVaR and Tail Gini of gains. If None it duplicates alpha value.
        The default is None.
    b_sim : float, optional
        Number of CVaRs used to approximate Tail Gini of gains. If None it duplicates a_sim value.
        The default is None.
    kappa : float, optional
        Deformation parameter of RLVaR and RLDaR, must be between 0 and 1. The default is 0.30.
    solver: str, optional
        Solver available for CVXPY that supports power cone programming. Used to calculate RLVaR and RLDaR.
        The default value is None.
    cmap : cmap, optional
        Colorscale that represents the risk adjusted return ratio.
        The default is 'viridis'.
    w : DataFrame of shape (n_assets, 1), optional
        A portfolio specified by the user. The default is None.
    label : str or list, optional
        Name or list of names of portfolios that appear on plot legend.
        The default is 'Portfolio'.
    marker : str, optional
        Marker of w. The default is "*".
    s : float, optional
        Size of marker. The default is 16.
    c : str, optional
        Color of marker. The default is 'r'.
    height : float, optional
        Height of the image in inches. The default is 6.
    width : float, optional
        Width of the image in inches. The default is 10.
    t_factor : float, optional
        Factor used to annualize expected return and expected risks for
        risk measures based on returns (not drawdowns). The default is 252.

        .. math::

            \begin{align}
            \text{Annualized Return} & = \text{Return} \, \times \, \text{t_factor} \\
            \text{Annualized Risk} & = \text{Risk} \, \times \, \sqrt{\text{t_factor}}
            \end{align}

    ax : matplotlib axis, optional
        If provided, plot on this axis. The default is None.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot for further tweaking.

    Example
    -------
    ::

        label = 'Max Risk Adjusted Return Portfolio'
        mu = port.mu
        cov = port.cov
        returns = port.returns

        ax = rp.plot_frontier(w_frontier=ws,
                              mu=mu,
                              cov=cov,
                              returns=returns,
                              rm=rm,
                              rf=0,
                              alpha=0.05,
                              cmap='viridis',
                              w=w1,
                              label=label,
                              marker='*',
                              s=16,
                              c='r',
                              height=6,
                              width=10,
                              t_factor=252,
                              ax=None)

    .. image:: images/MSV_Frontier.png


    """

    if not isinstance(w_frontier, pd.DataFrame):
        raise ValueError("w_frontier must be a DataFrame")

    if not isinstance(mu, pd.DataFrame):
        raise ValueError("mu must be a DataFrame")

    if not isinstance(cov, pd.DataFrame):
        raise ValueError("cov must be a DataFrame")

    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a DataFrame")

    if returns.shape[1] != w_frontier.shape[0]:
        a1 = str(returns.shape)
        a2 = str(w_frontier.shape)
        raise ValueError("shapes " + a1 + " and " + a2 + " not aligned")

    if w is not None:
        if not isinstance(w, pd.DataFrame):
            raise ValueError("w must be a DataFrame")

        if w.shape[1] > 1 and w.shape[0] == 1:
            w = w.T

        if returns.shape[1] != w.shape[0]:
            a1 = str(returns.shape)
            a2 = str(w.shape)
            raise ValueError("shapes " + a1 + " and " + a2 + " not aligned")

    if beta is None:
        beta = alpha
    if b_sim is None:
        b_sim = a_sim

    if ax is None:
        fig = plt.gcf()
        ax = fig.gca()
        fig.set_figwidth(width)
        fig.set_figheight(height)
    else:
        fig = ax.get_figure()

    mu_ = np.array(mu, ndmin=2)

    if kelly == False:
        ax.set_ylabel("Expected Arithmetic Return")
    elif kelly == True:
        ax.set_ylabel("Expected Logarithmic Return")

    item = rmeasures.index(rm)
    if rm in ["CVaR", "TG", "EVaR", "RLVaR", "CVRG", "TGRG", "CDaR", "EDaR", "RLDaR"]:
        x_label = (
            rm_names[item] + " (" + rm + ")" + " $\\alpha = $" + "{0:.2%}".format(alpha)
        )
    else:
        x_label = rm_names[item] + " (" + rm + ")"
    if rm in ["CVRG", "TGRG"]:
        x_label += ", $\\beta = $" + "{0:.2%}".format(beta)
    if rm in ["RLVaR", "RLDaR"]:
        x_label += ", $\\kappa = $" + "{0:.2}".format(kappa)
    ax.set_xlabel("Expected Risk - " + x_label)

    title = "Efficient Frontier Mean - " + x_label
    ax.set_title(title)

    X1 = []
    Y1 = []
    Z1 = []

    for i in range(w_frontier.shape[1]):
        try:
            weights = np.array(w_frontier.iloc[:, i], ndmin=2).T
            risk = rk.Sharpe_Risk(
                weights,
                cov=cov,
                returns=returns,
                rm=rm,
                rf=rf,
                alpha=alpha,
                a_sim=a_sim,
                beta=beta,
                b_sim=b_sim,
                kappa=kappa,
                solver=solver,
            )

            if kelly == False:
                ret = mu_ @ weights
            elif kelly == True:
                ret = 1 / returns.shape[0] * np.sum(np.log(1 + returns @ weights))
            ret = ret.item() * t_factor

            if rm not in ["MDD", "ADD", "CDaR", "EDaR", "RLDaR", "UCI"]:
                risk = risk * t_factor**0.5

            ratio = (ret - rf) / risk

            X1.append(risk)
            Y1.append(ret)
            Z1.append(ratio)
        except:
            pass

    ax1 = ax.scatter(X1, Y1, c=Z1, cmap=cmap)

    if w is not None:
        if isinstance(label, str):
            label = [label]

        if label is None:
            label = w.columns.tolist()

        if w.shape[1] != len(label):
            label = w.columns.tolist()

        label = [
            v + " " + str(label[:i].count(v) + 1) if label.count(v) > 1 else v
            for i, v in enumerate(label)
        ]

        if isinstance(c, str):
            colormap = np.array(colors.to_rgba(c)).reshape(1, -1)
        elif c is None:
            colormap = np.array(colors.to_rgba("red")).reshape(1, -1)

        elif isinstance(c, list):
            colormap = [list(colors.to_rgba(i)) for i in c]
            colormap = np.array(colormap)

        if len(label) != colormap.shape[0]:
            colormap = cm.get_cmap("tab20")
            colormap = colormap(np.linspace(0, 1, 20))
            colormap = np.vstack(
                [colormap[6:8], colormap[2:6], colormap[8:], colormap[0:2]]
            )

        n_repeats = int(len(label) // 20 + 1)
        if n_repeats > 1:
            colormap = np.vstack([colormap] * n_repeats)

        for i in range(w.shape[1]):
            weights = w.iloc[:, i].to_numpy().reshape(-1, 1)
            risk = rk.Sharpe_Risk(
                weights,
                cov=cov,
                returns=returns,
                rm=rm,
                rf=rf,
                alpha=alpha,
                a_sim=a_sim,
                beta=beta,
                b_sim=b_sim,
                kappa=kappa,
                solver=solver,
            )
            if kelly == False:
                ret = mu_ @ weights
            elif kelly == True:
                ret = 1 / returns.shape[0] * np.sum(np.log(1 + returns @ weights))
            ret = ret.item() * t_factor

            if rm not in ["MDD", "ADD", "CDaR", "EDaR", "RLDaR", "UCI"]:
                risk = risk * t_factor**0.5

            color = colormap[i].reshape(1, -1)
            ax.scatter(risk, ret, marker=marker, s=s**2, c=color, label=label[i])

        ax.legend(loc="upper left")

    xmin = np.min(X1) - np.abs(np.max(X1) - np.min(X1)) * 0.1
    xmax = np.max(X1) + np.abs(np.max(X1) - np.min(X1)) * 0.1
    ymin = np.min(Y1) - np.abs(np.max(Y1) - np.min(Y1)) * 0.1
    ymax = np.max(Y1) + np.abs(np.max(Y1) - np.min(Y1)) * 0.1

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    ax.xaxis.set_major_locator(plt.AutoLocator())

    ticks_loc = ax.get_yticks().tolist()
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(["{:.2%}".format(x) for x in ticks_loc])
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels(["{:.2%}".format(x) for x in ticks_loc])

    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    ax.grid(linestyle=":")

    #colorbar = ax.figure.colorbar(ax1)
    #colorbar.set_label("Risk Adjusted Return Ratio")

    try:
        fig.tight_layout()
    except:
        pass

    return ax