#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def compare_bode_bicycles(bikes, speed, u, y, fig=None):
    """Returns a figure with the Bode plots of multiple bicycles.

    Parameters
    ----------
    bikes : list
        A list of bicycleparameters.Bicycle instances.
    speed : float
        The speed at which to evaluate the system.
    u : integer
        An integer between 0 and 1 corresponding to the inputs roll torque
        and steer torque.
    y : integer
        An integer between 0 and 3 corresponding to the inputs roll rate,
        steer rate, roll angle and steer angle.

    Returns
    -------
    fig : matplotlib.Figure instance
        The Bode plot.

    Notes
    -----
    The phases are matched around zero degrees at with respect to the first
    frequency.

    """

    if fig is None:
        fig = plt.figure()

    for bike in bikes:
        bike.plot_bode(speed, u, y, label=bike.bicycleName, fig=fig)

    # take care of phase misalignment
    phaseLines = fig.ax2.lines
    for line in phaseLines:
        firstValue = line.get_ydata()[0]
        n = np.ceil(np.floor(abs(firstValue / 180.)) / 2.)
        line.set_ydata(line.get_ydata() - np.sign(firstValue) * n * 360.)

    return fig

def plot_eigenvalues(bikes, speeds, colors=None, linestyles=None,
        largest=False, show=False):
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

    if show is True:
        plt.show()

    return fig
