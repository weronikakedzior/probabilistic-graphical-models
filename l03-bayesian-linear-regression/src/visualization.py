import math

import torch
from matplotlib import pyplot as plt
import numpy as np


def _plot_data(ax, df, x, properties=dict(), regression=None):
    if 'category' not in properties:
        ax.scatter(
            x=df[x],
            y=df[properties.get('y', 'y')]
        )
    else:
        for cat in df[properties['category']].unique():
            mask = df[properties['category']] == cat
            ax.scatter(
                x=df[mask][x],
                y=df[mask][properties.get('y', 'y')],
                label=cat
            )

    if regression is not None:
        ax.scatter(df[x], regression, marker='+', c='k')

    ax.set_xlabel(properties.get('x_label', x))
    ax.set_ylabel(properties.get('y_label', properties.get('y', 'y')))
    if 'category' in properties:
        ax.legend()


def _get_regression_prediction(parameters, df):
    intercept = parameters.pop('intercept', 0)
    features = list(parameters.keys())
    X = df[features].to_numpy()
    beta = np.array([parameters[f] for f in features])
    return np.dot(X, beta) + intercept


def plot_data(df, properties=dict(), regression_parameters=dict()):
    if len(regression_parameters) > 0:
        regression = _get_regression_prediction(
            parameters=regression_parameters,
            df=df
        )
    else:
        regression = None
    if 'x' in properties:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        _plot_data(
            ax=ax,
            df=df,
            x=properties['x'],
            properties=properties,
            regression=regression,
        )
    else:
        real_features = [f for f in df.columns if f != properties.get('y', 'y')]
        if 'category' in properties:
            real_features = [
                f for f in real_features if f != properties['category']
            ]
        num_plots = len(real_features)
        fig, ax = plt.subplots(
            nrows=math.ceil(num_plots/3),
            ncols=3,
            figsize=(15, 5*math.ceil(num_plots/3))
        )
        ax = ax.ravel()
        for idx, x in enumerate(real_features):
            _plot_data(
                ax=ax[idx],
                x=x,
                df=df,
                properties=properties,
                regression=regression,
            )

    plt.show()


def plot_ols(data, properties=dict()):
    """Plots OLS."""

    FONT_SIZE = 14

    plt.rc('font', size=FONT_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONT_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title

    fig, axs = plt.subplots(
        nrows=2, ncols=1,
        figsize=(15, 18),
        sharey=True,
        sharex=True,
        squeeze=True,
    )

    x_col = properties.get('x', 0)
    x_label = properties.get('x_label', 'x')
    y_label = properties.get('y_label', 'y')

    cat_col = properties.get('category', None)

    for ax, (predictor_name, predictor) in zip(axs, data.items()):
        x = predictor['x']
        y = predictor['y']
        ols = predictor['ols']

        y_mean = ols[0]
        y_bottom_ci = ols[1][:, 0]
        y_top_ci = ols[1][:, 1]
        y_bottom_pi = ols[2][:, 0]
        y_top_pi = ols[2][:, 1]

        xplot, ym, ylbci, yubci, ylbpi, yubpi, y_true = list(zip(*sorted(
            zip(
                x[:, x_col].tolist(),
                y_mean,
                y_bottom_ci,
                y_top_ci,
                y_bottom_pi,
                y_top_pi,
                y,
            ),
            key=lambda r: r[0]
        )))

        ax.plot(
            xplot,
            ym,
            color="red",
            label="Mean output"
        )
        ax.fill_between(
            xplot,
            ylbpi,
            yubpi,
            color='cornflowerblue',
            alpha=0.5,
            label=f"Prediction Interval"
        )

        pi_ratio = np.sum(
            (torch.stack(ylbpi, dim=0).numpy() <= y_true) &
            (y_true <= torch.stack(yubpi, dim=0).numpy())
        ) / len(y_true)
        ax.fill_between(
            xplot,
            ylbci,
            yubci,
            color='orange',
            alpha=0.5,
            label=f"Confidence Interval"
        )
        ci_ratio = np.sum(
            (torch.stack(ylbci, dim=0).numpy() <= y_true) &
            (y_true <= torch.stack(yubci, dim=0).numpy())
        ) / len(y_true)

        ax.text(1, 0, f'Percent of observations within\n'
                      f'Confidence interval: {ci_ratio*100:.2f}%\n'
                      f'Prediction interval: {pi_ratio*100:.2f}%',
                horizontalalignment='right',
                verticalalignment='bottom', transform=ax.transAxes)
        if cat_col is None:
            ax.scatter(
                xplot, y_true,
                marker='+', s=100,
                alpha=1,
                c='green',
                label="True values"
            )
        else:
            for cat in np.unique(x[:, cat_col]):
                mask = x[:, cat_col] == cat
                ax.scatter(
                    np.array(xplot)[mask], np.array(y_true)[mask],
                    marker='+', s=100,
                    alpha=1,
                    label=cat
                )
        ax.set(
            xlabel=x_label,
            ylabel=y_label,
            title=f'{predictor_name} set',
        )

        ax.set_ylabel(f"{y_label}")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc='center right')
    fig.legend(handles[3:], labels[3:], loc='center left', title='Category')
    plt.show()
