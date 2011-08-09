#!/usr/bin/env python

import matplotlib.pyplot as plt

def plot_eigenvalues(bikes, speeds, colors=None, linestyles=None, largest=False):
    '''Returns a figure with the eigenvalues vs speed for multiple bicycles.

    Parameters
    ----------
    bikes : list
        A list of Bicycle objects.
    speeds : ndarray, shape(n,)
        An array of speeds.
    colors : list
        A list of matplotlib colors for each bicycle.
    linestyles : list
        A list of matplotlib linestyles for each bicycle.
    largest : boolean
        If true, only plots the largest eigenvalue.

    Returns
    -------
    fig : matplotlib figure

    '''

    numBikes = len(bikes)

    if not colors:
        # define some colors for the parts
        cmap = plt.get_cmap('gist_rainbow')
        colors = []
        for i in range(numBikes):
            colors.append(cmap(1. * i / numBikes))

    if not linestyles:
        linestyles = ['-'] * numBikes

    for i, bike in enumerate(bikes):
        if i == 0:
            fig = None
        fig = bike.plot_eigenvalues_vs_speed(speeds, fig=fig, color=colors[i],
                                             largest=largest, generic=True,
                                             linestyle=linestyles[i],
                                             show=False)
        plt.legend()
        plt.axis('tight')

    plt.show()

    return fig
