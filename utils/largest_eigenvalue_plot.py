import bicycleparameters as bp
import numpy as np

bikeNames = ['Browserins', 'Browser', 'Pista',
             'Fisher', 'Yellow', 'Yellowrev']

colors = ['k'] * 3 + ['gray'] * 3

linestyles = ['-', '-.', '--'] * 2

bikes = []

for name in bikeNames:
    bikes.append(bp.Bicycle(name, pathToData='../data'))

speeds = np.linspace(0., 10., num=100)

bp.plot_eigenvalues(bikes, speeds, colors=colors,
                    linestyles=linestyles), largest=True)

bp.plt.vlines([2.5, 5., 7.], -0.5, 4.)

#bp.plt.savefig('../plots/largest.png')

bp.plt.show()
