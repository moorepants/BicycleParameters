import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from .geometry import fundamental_geometry_plot_data


class BenchmarkParameterSet(object):
    """Represents the parameters of the benchmark bicycle presented in
    Meijaard2007."""

    # maps "Python" string to LaTeX version
    par_strings = {
        'IBxx': r'I_{Bxx}',
        'IBxz': r'I_{Bxz}',
        'IByy': r'I_{Byy}',
        'IBzz': r'I_{Bzz}',
        'IFxx': r'I_{Fxx}',
        'IFyy': r'I_{Fyy}',
        'IHxx': r'I_{Hxx}',
        'IHxz': r'I_{Hxz}',
        'IHyy': r'I_{Hyy}',
        'IHzz': r'I_{Hzz}',
        'IRxx': r'I_{Rxx}',
        'IRyy': r'I_{Ryy}',
        'c': r'c',
        'g': r'g',
        'lam': r'\lambda',
        'mB': r'm_B',
        'mF': r'm_F',
        'mH': r'm_H',
        'mR': r'm_R',
        'rF': r'r_F',
        'rR': r'r_R',
        'v': r'v',
        'w': r'w',
        'xB': r'x_B',
        'xH': r'x_H',
        'zB': r'z_B',
        'zH': r'z-H',
    }

    def __init__(self, parameters, includes_rider):
        """Initializes a parameter set based on Meijaard2007.

        Parameters
        ==========
        parameters : dictionary
            A dictionary mapping variable names to values.
        includes_rider : boolean
            True if body B is the combined rear frame and rider.

        """
        self.parameters = parameters
        self.includes_rider = includes_rider
        numColors = 4
        cmap = plt.get_cmap('gist_rainbow')
        self.part_colors = {}
        for i, part in enumerate(['B', 'H', 'R', 'F']):
            self.part_colors[part] = cmap(1. * i / numColors)

    def plot_geometry(self, show_steer_axis=True, ax=None):
        """Returns a matplotlib axes with the simplest drawing of the bicycle's
        geometry.

        Parameters
        ==========
        show_steer_axis : boolean
            If true, a dotted line will be plotted along the steer axis from
            the front wheel center to the ground.
        ax : AxesSubplot, optional
            An axes to draw on, otherwise one is created.

        """

        p = self.parameters

        if ax is None:
            fig, ax = plt.subplots()

        # plot the ground line
        ax.axhline(0.0, color='black')

        # plot the rear wheel
        c = patches.Circle((0., p['rR']), radius=p['rR'],
                           linewidth=2, fill=False)
        ax.add_patch(c)

        # plot the front wheel
        c = patches.Circle((p['w'], p['rF']), radius=p['rF'],
                           linewidth=2, fill=False)
        ax.add_patch(c)

        # plot the fundamental bike
        deex, deez = fundamental_geometry_plot_data(p)
        ax.plot(deex, -deez, 'k', linewidth=2)

        if show_steer_axis:
            # plot the steer axis
            dx3 = deex[2] + deez[2] * (deex[2] - deex[1]) / (deez[1] - deez[2])
            ax.plot([deex[2], dx3],  [-deez[2], 0.], 'k--')

        ax.set_aspect('equal')

        return ax

    def plot_principal_radii_of_gyration(self, ax=None):

        p = self.parameters

        if ax is None:
            fig, ax = plt.subplots()

        def plot_titled_radii(b):

            color = self.part_colors[b]

            x = 'x{}'.format(b)
            z = 'z{}'.format(b)

            angle_max, radius_max, angle_min, radius_min = \
                self._planar_principal_radii_of_gyration(b)

            ax.plot([p[x], p[x] + np.cos(angle_max) * radius_min],
                    [-p[z], -p[z] - np.sin(angle_max) * radius_min],
                    color=color)
            ax.plot([p[x], p[x] - np.cos(angle_max) * radius_min],
                    [-p[z], -p[z] + np.sin(angle_max) * radius_min],
                    color=color)

            ax.plot([p[x], p[x] + np.cos(angle_min) * radius_max],
                    [-p[z], -p[z] - np.sin(angle_min) * radius_max],
                    color=color)
            ax.plot([p[x], p[x] - np.cos(angle_min) * radius_max],
                    [-p[z], -p[z] + np.sin(angle_min) * radius_max],
                    color=color)

            kyy = np.sqrt(p['I{}yy'.format(b)] / p['m{}'.format(b)])

            c = patches.Circle((p[x], -p[z]), radius=kyy, fill=False,
                               color=color)
            ax.add_patch(c)

        plot_titled_radii('H')
        plot_titled_radii('B')

        kRxx = np.sqrt(p['IRxx'] / p['mR'])
        kRyy = np.sqrt(p['IRyy'] / p['mR'])
        ax.plot([-kRxx, kRxx], [p['rR'], p['rR']], color=self.part_colors['R'])
        ax.plot([0.0, 0.0], [p['rR'] - kRxx, p['rR'] + kRxx],
                color=self.part_colors['R'])
        c = patches.Circle((0., p['rR']), radius=kRyy,
                           fill=False, color=self.part_colors['R'])
        ax.add_patch(c)

        kFxx = np.sqrt(p['IFxx'] / p['mF'])
        kFyy = np.sqrt(p['IFyy'] / p['mF'])
        ax.plot([p['w']-kFxx, p['w']+kFxx], [p['rF'], p['rF']],
                color=self.part_colors['F'])
        ax.plot([p['w'], p['w']], [p['rF'] - kFxx, p['rF'] + kFxx],
                color=self.part_colors['F'])
        c = patches.Circle((p['w'], p['rF']), radius=kFyy,
                           fill=False, color=self.part_colors['F'])
        ax.add_patch(c)

        ax.set_aspect('equal')

        return ax

    def _planar_principal_radii_of_gyration(self, body):

        p = self.parameters
        b = body

        IH = np.array([[p['I{}xx'.format(b)], p['I{}xz'.format(b)]],
                       [p['I{}xz'.format(b)], p['I{}zz'.format(b)]]])

        evals, evecs = np.linalg.eig(IH)

        idxs = np.argsort(evals)

        evals = evals[idxs]
        evecs = evecs[:, idxs]

        angle_to_max = np.arctan2(evecs[1, 0], evecs[0, 0])
        angle_to_min = np.arctan2(evecs[1, 1], evecs[0, 1])

        kHmax = np.sqrt(evals[0] / p['m{}'.format(b)])
        kHmin = np.sqrt(evals[1] / p['m{}'.format(b)])

        return angle_to_max, kHmax, angle_to_min, kHmin

    def plot_mass_centers(self, ax=None):
        """Returns a Matplotlib axes with each of the four mass centers marked.

        Parameters
        ==========
        ax : AxesSubplot, optional
            An axes to draw on, otherwise one is created.

        """

        p = self.parameters

        if ax is None:
            fig, ax = plt.subplots()

        def com_symbol(ax, center, radius, color='b'):
            '''Returns axis with center of mass symbol.'''
            c = patches.Circle(center, radius=radius, fill=False)
            w1 = patches.Wedge(center, radius, 0., 90.,
                               color=color, ec=None, alpha=0.5)
            w2 = patches.Wedge(center, radius, 180., 270.,
                               color=color, ec=None, alpha=0.5)
            ax.add_patch(w1)
            ax.add_patch(w2)
            ax.add_patch(c)
            return ax

        # radius of the CoM symbol
        sRad = 0.03

        # front wheel CoM
        ax = com_symbol(ax, (p['w'], p['rF']), sRad,
                        color=self.part_colors['F'])
        ax.text(p['w'] + sRad, p['rF'] + sRad, 'F')
        # rear wheel CoM
        ax = com_symbol(ax, (0., p['rR']), sRad, color=self.part_colors['R'])
        ax.text(0. + sRad, p['rR'] + sRad, 'R')
        # front frame CoM
        ax = com_symbol(ax, (p['xH'], -p['zH']), sRad,
                        color=self.part_colors['H'])
        ax.text(p['xH'] + sRad, -p['zH'] + sRad, 'H')
        # rear frame (and rider) CoM
        ax = com_symbol(ax, (p['xB'], -p['zB']), sRad,
                        color=self.part_colors['B'])
        s = 'B (frame + rider)'
        ax.text(p['xB'] + sRad, -p['zB'] + sRad,
                s if self.includes_rider else 'B')

        ax.set_aspect('equal')

        return ax

    def plot_inertia_ellipsoids(self, ax=None):
        """Returns a Matplotlib axes with 2D representations of 3D solid
        uniform ellipsoids that have the same inertia as the body.

        Parameters
        ==========
        ax : AxesSubplot, optional
            An axes to draw on, otherwise one is created.

        """
        p = self.parameters

        if ax is None:
            fig, ax = plt.subplots()

        def add_ellipse(b):
            angle_max, radius_max, angle_min, radius_min = \
                self._planar_principal_radii_of_gyration(b)
            kyy = np.sqrt(p['I{}yy'.format(b)] / p['m{}'.format(b)])

            width = np.sqrt(5/2*(-radius_max**2 + kyy**2 + radius_min**2))
            height = np.sqrt(5/2*(radius_max**2 + kyy**2 - radius_min**2))
            ellipse = patches.Ellipse((p['x{}'.format(b)],
                                       -p['z{}'.format(b)]), width, height,
                                      angle=-np.rad2deg(angle_max), fill=False,
                                      color=self.part_colors[b], alpha=0.25)
            ax.add_patch(ellipse)

        add_ellipse('H')
        add_ellipse('B')

        return ax
