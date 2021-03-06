import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.legend import Legend


def plot_bars_with_subplots(nrows, ncols, x_values, y_values, colors, labels):
    """
    Create a plot for areas and populations of each city in the dataset

    :param nrows: number of rows subplot
    :param ncols: number of cols subplot
    :param x_values: list of x values
    :param y_values: list of lists -- data to plot (one list for each subplot)
    :param colors: list of colors to use for each parameter to represent
    :param labels: labels to use (list)
    :return: fig, axs
    """

    fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(12, 9))

    for ax, color, y_values, label in zip(axs, colors, y_values, labels):
        ax.bar(x_values, y_values, label=label, color=color)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.legend()

    plt.xticks(rotation=70)

    return fig


def plot_measures_with_subplots(nrows, ncols, x_values, y_values, colors, labels):
    """
    Plot info passed as parameters in a single figure with possible sublplots

    :param nrows: number of rows in subplots
    :param ncols: number of columns in subplots
    :param x_values: np.array of x_values
    :param y_values: lis of lists of y_values
    :param colors: list of colors
    :param labels: list of strings (labels)
    :return: fig, axs
    """

    fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(12, 9))

    for ax, color, y_values, label in zip(axs, colors, y_values, labels):
        ax.plot(x_values, y_values, color=color, marker='o')
        ax.set_ylabel(label)

    plt.xticks(rotation=70)

    return fig, axs


def plot_measures(df):
    """
    plot dataframe of measures in multiple subplots

    :param df: pandas dataframe of basic measures
    :return: fig
    """

    x_values = df.columns.values
    y_values = df.values.tolist()
    colors = ['r', 'b', 'g', 'y', 'm']
    # labels = df.index.values
    labels = ['N', 'E', 'density', 'diameter', '<c>']

    fig, axs = plot_measures_with_subplots(len(labels), 1, x_values, y_values, colors, labels)
    # fig.legend()

    return fig


def create_scatter(x_values, y_values, x_label, y_label, labels, colors, areas=None):
    """
    Creates a scatter plot of y_values as a function of x_values.

    :param x_values: np.array
    :param y_values: np.array
    :param x_label: string
    :param y_label: string - a generic label of the x axis
    :param labels: list of strings - labels of scatter plots
    :param colors: list of strings
    :param areas: dimension of scatter points - list

    :return: fig
    """

    fig, ax = plt.subplots(figsize=(8, 6))
    dots = []
    style = dict(size=10, color='gray')

    if areas is None:
        for x_value, y_value, color, label in zip(x_values, y_values, colors, labels):
            dots.append(ax.scatter(x_value, y_value, color=color, label=label, alpha=0.5))

        max_x = max(x_values)
        max_y = max(y_values)
        for i, txt in enumerate(labels):
            if x_values[i] > 1/3*max_x or y_values[i] > 1/3*max_y:
                v_align = 'top' if labels[i] == 'antofagasta' or labels[i] == 'rennes' else 'bottom'
                ax.annotate(txt, (x_values[i], y_values[i]), ha='center', verticalalignment=v_align)

    else:
        for x_value, y_value, color, label, area in zip(x_values, y_values, colors, labels, areas):
            dots.append(ax.scatter(x_value, y_value, color=color, label=label, s=area, alpha=0.5))

        max_area = max(areas)
        for i, txt in enumerate(labels):
            if areas[i] > 1/2*max_area:
                ax.annotate(txt, (x_values[i], y_values[i]), ha='center')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.2)
    plt.subplots_adjust(top=0.75)

    leg = Legend(ax, dots[:3], ['Oceania', 'America', 'Europe'], loc='best', frameon=False)
    for handle in leg.legendHandles:
        handle.set_sizes([25.0])
    ax.add_artist(leg)

    return fig


