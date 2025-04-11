from abc import ABC
import os

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from dtk.bicycle import benchmark_to_moore

from .com import total_com
from .geometry import fundamental_geometry_plot_data
from .conversions import _convert_principal_to_benchmark

__all__ = [
    'Meijaard2007ParameterSet',
    'Meijaard2007WithFeedbackParameterSet',
    'Moore2012ParameterSet',
    'Moore2019ParameterSet',
    'MooreRiderLean2012ParameterSet',
    'ParameterSet',
]


def _com_symbol(ax, center, radius, color='b', label=None):
    """Returns matplotlib axis with center of mass symbol and possibly a label
    added."""
    c = patches.Circle(center, radius=radius, fill=False)
    w1 = patches.Wedge(center, radius, 0., 90., color=color, ec=None,
                       alpha=0.5)
    w2 = patches.Wedge(center, radius, 180., 270., color=color, ec=None,
                       alpha=0.5)
    ax.add_patch(w1)
    ax.add_patch(w2)
    ax.add_patch(c)

    if label is not None:
        ax.text(center[0] + radius, center[1] + radius, label)

    return ax


def _plot_geometry_from_benchmark(ax, parameters, show_steer_axis=True):
    """Returns a matplotlib axis with the basic bicycle geometry plotted."""

    p = parameters

    # plot the ground line
    ax.axhline(0.0, color='black')

    # plot the rear wheel
    c = patches.Circle((0., -p['rR']), radius=p['rR'],
                       linewidth=2, fill=False)
    ax.add_patch(c)

    # plot the front wheel
    c = patches.Circle((p['w'], -p['rF']), radius=p['rF'],
                       linewidth=2, fill=False)
    ax.add_patch(c)

    # plot the fundamental bike (Moore2012 depiction)
    deex, deez = fundamental_geometry_plot_data(p)
    ax.plot(deex, deez, color='k', linewidth=2)

    if show_steer_axis:
        # plot the steer axis
        dx3 = deex[2] - deez[2] * (deex[2] - deex[1]) / (-deez[1] + deez[2])
        ax.plot([deex[2], dx3], [deez[2], 0.], 'k--')

    return ax


class ParameterSet(ABC):
    """A parameter set is a collection of constants with associated floating
    point values that are present in a set of differential algebraic equations
    that represent a multibody bicycle model. These pairs are typically defined
    in a specific academic article, dissertation, book chapter, or section and
    subclasses should be named in a way that ties them to that written work.
    Parameter sets must be named with this pattern ``NameOfMySetParameterSet``
    where ``NameOfMySet`` is a unique name other than something that includes
    ``ParameterSet``.

    A parameter set, or a subset of the parameters, can be used with multiple
    different models. A parameter set is associated with a particular
    parameterization of one or more models. Parameter sets can be converted to
    equivalent parameter sets, but only by assuming a particular model
    configuration. The obvious configuration for a bicycle model is the
    upright, zero steer state. But if, for example, a rider configuration is
    included, then some nominal configuration would need to be defined for
    conversion consistency.

    Each parameter set should have a unique set of ASCII strings that represent
    the constants in a model.

    Attributes
    ==========
    par_strings : dictionary
        Maps ASCII strings to their LaTeX string.

    """
    par_strings = {
        'aR': r'a_R',
        'beta': r'\beta',
    }

    def _check_parameters(self, parameters):
        """Ensures that each parameter in par_strings is present in parameters
        and that the values are floats."""
        for k, _ in self.par_strings.items():
            if k not in parameters.keys():
                msg = '{} is missing from the provided parameter dictionary.'
                raise ValueError(msg.format(k))
            if not isinstance(parameters[k], float):
                msg = '{} is not a valid value for parameter {}'
                raise ValueError(msg.format(parameters[k], k))

    def __init__(self, par_dict):
        self._check_parameters(par_dict)
        cls_name = self.__class__.__name__
        self.parameterization = cls_name.split('ParameterSet')[0]

    @classmethod
    def _from_yaml(cls, path_to_file):
        """Instantiates a ParameterSet from a file."""
        # TODO : Wouldn't this have to dynamically create the class so that the
        # class name is derived from a specifier in the yaml file?
        with open(os.path.join(path_to_file), 'r') as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        return cls(yaml_data['values'])

    def _generate_body_colors(self):
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        self.body_colors = {}
        for i, b in enumerate(self.body_labels):
            self.body_colors[b] = colors[i]

    def _invert_yaxis(self, ax):
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        return ax

    def _finalize_plot(self, ax):
        ax = self._invert_yaxis(ax)
        ax.set_aspect('equal')
        ax.autoscale()  # needed for the plots with only patches
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$z$ [m]')

    def to_parameterization(self, name):
        """Returns a specific parameter set based on the provided
        parameterization name.

        Parameters
        ==========
        name : string
            The name of the parameterization. These should correspond to a
            subclass of a ``ParameterSet`` and the name will be the string that
            precedes "ParameterSet". For example, the parameterization name of
            ``Meijaard2007ParameterSet`` is ``Meijaard2007``.

        Returns
        =======
        ParmeterSet
            If a different parameterization is requested and this class can
            convert itself, it will return a new parameter set of the correct
            parameterization.

        """
        if name == self.parameterization:
            return self
        else:
            msg = '{} is not an avaialble parameter set.'
            raise ValueError(msg.format(name))

    def to_yaml(self, fname):
        """Writes parameters to file in the YAML format.

        Parameters
        ==========
        fname : string
            Path to file.

        """
        data = {
            'description': '',
            'parameterization': self.parameterization,
            'parameters': '',
            'rider': None,
            'values': self.parameters,
        }
        with open(fname, 'w') as f:
            yaml.dump(data, f)

    def to_ini(self, fname):
        """Writes parameters to file in the INI format. Metadata is not
        included.

        Parameters
        ==========
        fname : string
            Path to file.

        """
        text = ""
        for k, v in self.parameters.items():
            text += "{}={:1.16f}\n".format(k, v)
        with open(fname, 'w') as f:
            f.write(text)

    def _repr_html_(self):
        opening = (r'<table><caption>{}</caption><tr><th>Variable</th>'
                   r'<th>Value</th></tr>')
        lines = opening.format(self.parameterization)
        for k, v in self.par_strings.items():
            latex_var = r'\({}\)'.format(v)
            row_str = r'<tr><td>{}</td><td>{:1.3f}</td></tr>'
            lines += row_str.format(latex_var, self.parameters[k])
        lines += r'</table>'
        return lines


class Meijaard2007ParameterSet(ParameterSet):
    """Represents the parameters of the benchmark bicycle presented in
    [Meijaard2007]_.

    The four bodies are:

    - B: rear frame + rigid rider
    - F: front wheel
    - H: front frame (fork & handlebars)
    - R: rear wheel

    Parameters
    ==========
    parameters : dictionary
        A dictionary mapping variable names to values that contains the
        following keys:

        - ``IBxx`` : x moment of inertia of the frame/rider [kg*m**2]
        - ``IBxz`` : xz product of inertia of the frame/rider [kg*m**2]
        - ``IBzz`` : z moment of inertia of the frame/rider [kg*m**2]
        - ``IFxx`` : x moment of inertia of the front wheel [kg*m**2]
        - ``IFyy`` : y moment of inertia of the front wheel [kg*m**2]
        - ``IHxx`` : x moment of inertia of the handlebar/fork [kg*m**2]
        - ``IHxz`` : xz product of inertia of the handlebar/fork [kg*m**2]
        - ``IHzz`` : z moment of inertia of the handlebar/fork [kg*m**2]
        - ``IRxx`` : x moment of inertia of the rear wheel [kg*m**2]
        - ``IRyy`` : y moment of inertia of the rear wheel [kg*m**2]
        - ``c`` : trail [m]
        - ``g`` : acceleration due to gravity [m/s**2]
        - ``lam`` : steer axis tilt [rad]
        - ``mB`` : frame/rider mass [kg]
        - ``mF`` : front wheel mass [kg]
        - ``mH`` : handlebar/fork assembly mass [kg]
        - ``mR`` : rear wheel mass [kg]
        - ``rF`` : front wheel radius [m]
        - ``rR`` : rear wheel radius [m]
        - ``w`` : wheelbase [m]
        - ``xB`` : x distance to the frame/rider center of mass [m]
        - ``xH`` : x distance to the frame/rider center of mass [m]
        - ``zB`` : z distance to the frame/rider center of mass [m]
        - ``zH`` : z distance to the frame/rider center of mass [m]

    includes_rider : boolean
        True if body B is the combined rear frame and rider in terms of
        mass and inertia values.

    Attributes
    ==========
    par_strings : dictionary
        Maps ASCII strings to their LaTeX string.
    body_labels : list of strings
        Single capital letters that correspond to the four rigid bodies in the
        model.

    References
    ==========

    .. [Meijaard2007] Meijaard J.P, Papadopoulos Jim M, Ruina Andy and Schwab
       A.L, 2007, Linearized dynamics equations for the balance and steer of a
       bicycle: a benchmark and review, Proc. R. Soc. A., 463:1955–1982
       http://doi.org/10.1098/rspa.2007.1857

    """

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
        'zH': r'z_H',
    }

    body_labels = ['B', 'F', 'H', 'R']

    def __init__(self, parameters, includes_rider):
        super().__init__(parameters)
        self.parameters = parameters
        self.includes_rider = includes_rider
        self._generate_body_colors()

    def to_parameterization(self, name):
        """Returns a specific parameter set based on the provided
        parameterization name.

        Parameters
        ==========
        name : string
            The name of the parameterization. These should correspond to a
            subclass of a ``ParameterSet`` and the name will be the string that
            precedes "ParameterSet". For example, the parameterization name of
            ``Meijaard2007ParameterSet`` is ``Meijaard2007``.

        Returns
        =======
        ParmeterSet
            If a different parameterization is requested and this class can
            convert itself, it will return a new parameter set of the correct
            parameterization.

        """
        if name == self.parameterization:
            return self
        elif name == 'Moore2012':
            moore_par = benchmark_to_moore(self.parameters)
            moore_par['v'] = self.parameters['v']
            return Moore2012ParameterSet(moore_par, self.includes_rider)
        elif name == 'Meijaard2007WithFeedback':
            par = self.parameters.copy()
            gain_names = ['kTphi_phi', 'kTphi_del', 'kTphi_phid', 'kTphi_deld',
                          'kTdel_phi', 'kTdel_del', 'kTdel_phid', 'kTdel_deld']
            for gain in gain_names:
                par[gain] = 0.0
            return Meijaard2007WithFeedbackParameterSet(par,
                                                        self.includes_rider)
        else:
            msg = '{} is not an avaialble parameter set'
            raise ValueError(msg.format(name))

    @classmethod
    def _from_yaml(cls, path_to_file):

        with open(os.path.join(path_to_file), 'r') as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
            benchmark_par = yaml_data['values']
            rider = yaml_data['rider']

        return cls(benchmark_par, rider)

    def _calc_derived_params(self):
        # These parameters are needed but are not specified by the user.
        p = self.parameters

        pext = {}
        pext['IFxz'] = 0.0
        pext['IFzz'] = p['IFxx']
        pext['IRxz'] = 0.0
        pext['IRzz'] = p['IRxx']
        pext['xF'] = p['w']
        pext['xR'] = 0.0
        pext['yB'] = 0.0
        pext['yF'] = 0.0
        pext['yH'] = 0.0
        pext['yR'] = 0.0
        pext['zF'] = -p['rF']
        pext['zR'] = -p['rR']

        return pext

    def form_mass_center_vector(self, body):
        """Returns an array representing the 3D vector to the mass center of
        the body from the origin at the rear wheel contact point.

        Parameters
        ==========
        body : string
            One of 'B', 'F', 'H', 'R'.

        Returns
        =======
        ndarray, shape(3,)
            A vector containing the X, Y, and X coordinates of the mass center
            of the body.

        Examples
        ========

        >>> from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
        >>> from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
        >>> p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
        >>> p.form_mass_center_vector('B')
        array([ 0.28909943,  0.        , -1.04029228])

        """

        p = self.parameters.copy()
        p.update(self._calc_derived_params())

        x = p['x{}'.format(body)]
        y = p['y{}'.format(body)]
        z = p['z{}'.format(body)]

        return np.array([x, y, z])

    def mass_center_of(self, *bodies):
        """Returns the vector locating the center of mass of the collection of
        bodies.

        Parameters
        ==========
        bodies : iterable of strings
            One or more of the ``body_labels``.

        Returns
        =======
        com : ndarray, shape(3,)
            Vector locating the center of mass of the bodies givien in
            ``bodies``.

        Examples
        ========

        >>> from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
        >>> from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
        >>> p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
        >>> p.mass_center_of('B', 'H')
        array([ 0.31096918,  0.        , -1.02923892])

        """
        if len(bodies) == 1:
            return self.form_mass_center_vector(bodies[0])
        else:
            coordinates = []
            masses = []
            for body in bodies:
                masses.append(self.parameters['m{}'.format(body)])
                coordinates.append(
                    self.form_mass_center_vector(body))

            coordinates = np.array(coordinates).T

            mass, com = total_com(coordinates, masses)

            return com

    def form_inertia_tensor(self, body):
        """Returns the inertia tensor with respect to the global coordinate
        system and the body's mass center.

        Parameters
        ==========
        body : string
            One of the ``body_labels``.

        Returns
        =======
        inertia_tensor : ndarray, shape(3, 3)
            Inertia tensor of the body with respect to the body's mass center
            and the model's coordinate system.

        Examples
        ========

        >>> from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
        >>> from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
        >>> p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
        >>> p.form_inertia_tensor('H')
        array([[ 0.25337959,  0.        , -0.07204524],
               [ 0.        ,  0.24613881,  0.        ],
               [-0.07204524,  0.        ,  0.09557708]])

        """

        p = self.parameters.copy()
        p.update(self._calc_derived_params())

        Ixx = p['I{}xx'.format(body)]
        Ixz = p['I{}xz'.format(body)]
        Iyy = p['I{}yy'.format(body)]
        Izz = p['I{}zz'.format(body)]

        inertia_tensor = np.array([[Ixx, 0.0, Ixz],
                                   [0.0, Iyy, 0.0],
                                   [Ixz, 0.0, Izz]])

        return inertia_tensor

    def plot_geometry(self, show_steer_axis=True, ax=None):
        """Returns a matplotlib axes with a simple drawing of the bicycle's
        geometry.

        Parameters
        ==========
        show_steer_axis : boolean
            If true, a dotted line will be plotted along the steer axis from
            the front wheel center to the ground.
        ax : AxesSubplot, optional
            An axes to draw on, otherwise one is created.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
           from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
           p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
           p.plot_geometry()

        """

        p = self.parameters

        if ax is None:
            fig, ax = plt.subplots()

        ax = _plot_geometry_from_benchmark(ax, p,
                                           show_steer_axis=show_steer_axis)

        self._finalize_plot(ax)

        return ax

    def plot_mass_centers(self, bodies=None, ax=None):
        """Returns a matplotlib axes with mass center indicators for each body.

        Parameters
        ==========
        bodies: list of strings, optional
            A subset of the strings present in the class attribute
            ``body_labels``.
        ax: matplotlib Axes, optional
            An axes to plot on.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
           from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
           p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
           p.plot_mass_centers()

        """

        if ax is None:
            fig, ax = plt.subplots()

        if bodies is None:
            bodies = self.body_labels

        for b in bodies:
            ax = self.plot_body_mass_center(b, ax=ax)

        self._finalize_plot(ax)

        return ax

    def plot_body_mass_center(self, body, ax=None):
        """Returns a matplotlib axes with a mass center symbol for the
        specified body to the plot.

        Parameters
        ==========
        body : string
            The body string: ``F``, ``H``, ``B``, or ``R``.
        ax : SubplotAxes, optional
            Axes to plot on.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
           from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
           p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
           p.plot_body_mass_center('B')

        """

        if ax is None:
            fig, ax = plt.subplots()

        p = self.parameters.copy()
        p.update(self._calc_derived_params())

        # mass center
        x = p['x{}'.format(body)]
        z = p['z{}'.format(body)]
        radius = p['w'] / 10
        ax = _com_symbol(ax, (x, z), radius, color=self.body_colors[body],
                         label=body)
        self._finalize_plot(ax)

        return ax

    def _planar_principal_radii_of_gyration(self, body):

        p = self.parameters.copy()
        p.update(self._calc_derived_params())

        IH = self.form_inertia_tensor(body)
        # remove the Y row and column
        IH_xz_plane = np.delete(np.delete(IH, 1, axis=1), 1, axis=0)

        evals, evecs = np.linalg.eig(IH_xz_plane)

        idxs = np.argsort(evals)

        # NOTE : min is first entry, max is second entry
        evals = evals[idxs]
        evecs = evecs[:, idxs]

        # NOTE : The negative sign on the z value ensures the sign of the
        # rotation about the Y axis is correct, i.e. arctan2 is thinking you
        # are rotating in XY plane about Z.
        angle_to_max = np.arctan2(-evecs[1, 1], evecs[0, 1])

        kmin = np.sqrt(evals[0] / p['m{}'.format(body)])
        kmax = np.sqrt(evals[1] / p['m{}'.format(body)])
        kyy = np.sqrt(p['I{}yy'.format(body)] / p['m{}'.format(body)])

        return kmax, kmin, kyy, angle_to_max

    def plot_body_principal_radii_of_gyration(self, body, ax=None):
        """Returns a matplotlib axes with lines and a circle that indicate the
        principal radii of gyration of the specified body.

        Parameters
        ==========
        body : string
            One of the ``body_labels``.
        ax : SubplotAxes, optional
            Axes to plot on.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
           from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
           p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
           p.plot_body_principal_radii_of_gyration('B')

        """
        if ax is None:
            fig, ax = plt.subplots()

        p = self.parameters.copy()
        p.update(self._calc_derived_params())

        x = p['x{}'.format(body)]
        z = p['z{}'.format(body)]
        kaa, kbb, kyy, alpha = self._planar_principal_radii_of_gyration(body)

        c = patches.Circle((x, z), radius=kyy, fill=False,
                           color=self.body_colors[body], linestyle='--')
        ax.add_patch(c)

        # NOTE : -alpha is required because we are mapping the xz axes to a new
        # planar drawing grid which is x and y with z pointing out of the
        # screen
        ax.plot([x - kbb*np.cos(-alpha), x + kbb*np.cos(-alpha)],
                [z - kbb*np.sin(-alpha), z + kbb*np.sin(-alpha)],
                color=self.body_colors[body], linestyle="--")

        ax.plot([x - kaa*np.cos(-alpha - np.pi/2),
                 x + kaa*np.cos(-alpha - np.pi/2)],
                [z - kaa*np.sin(-alpha - np.pi/2),
                 z + kaa*np.sin(-alpha - np.pi/2)],
                color=self.body_colors[body], linestyle="--")

        self._finalize_plot(ax)

        return ax

    def plot_principal_radii_of_gyration(self, bodies=None, ax=None):
        """Returns a matplotlib axis with principal radii of all bodies
        shown.

        Parameters
        ==========
        bodies: list of strings, optional
            A subset of the strings present in the class attribute
            ``body_labels``.
        ax: matplotlib Axes, optional
            An axes to plot on.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
           from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
           p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
           p.plot_principal_radii_of_gyration()

        """
        if ax is None:
            fig, ax = plt.subplots()

        if bodies is None:
            bodies = self.body_labels

        for b in bodies:
            ax = self.plot_body_principal_radii_of_gyration(b, ax=ax)

        self._finalize_plot(ax)

        return ax

    def plot_body_principal_inertia_ellipsoid(self, body, ax=None):
        """Returns a matplotlib axes with an ellipse that respresnts the XZ
        plane view of a constant density ellipsoid which has the same principal
        moments and axes of inertia as the body.

        Parameters
        ==========
        body : string
            One of the ``body_labels``.
        ax : SubplotAxes, optional
            Axes to plot on.


        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
           from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
           p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
           p.plot_body_principal_inertia_ellipsoid('H')

        """

        if ax is None:
            fig, ax = plt.subplots()

        p = self.parameters.copy()
        p.update(self._calc_derived_params())

        kaa, kbb, kyy, alpha = self._planar_principal_radii_of_gyration(body)

        width = np.sqrt(5/2*(-kaa**2 + kyy**2 + kbb**2))
        height = np.sqrt(5/2*(kaa**2 + kyy**2 - kbb**2))

        ellipse = patches.Ellipse((p['x{}'.format(body)], p['z{}'.format(body)]),
                                  width, height,
                                  angle=-np.rad2deg(alpha), fill=False,
                                  color=self.body_colors[body])
        ax.add_patch(ellipse)

        self._finalize_plot(ax)

        return ax

    def plot_principal_inertia_ellipsoids(self, bodies=None, ax=None):
        """Returns a Matplotlib axes with 2D representations of 3D solid
        uniform ellipsoids that have the same inertia as the body.

        Parameters
        ==========
        bodies: list of strings, optional
            A subset of the strings present in the class attribute
            ``body_labels``.
        ax : AxesSubplot, optional
            An axes to draw on, otherwise one is created.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
           from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
           p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
           p.plot_principal_inertia_ellipsoids()

        """
        if ax is None:
            fig, ax = plt.subplots()

        if bodies is None:
            bodies = self.body_labels

        for b in bodies:
            ax = self.plot_body_principal_inertia_ellipsoid(b, ax=ax)

        self._finalize_plot(ax)

        return ax

    def plot_all(self, ax=None):
        """Returns matplotlib axes with the geometry and inertial
        representations of all bodies of the bicycle parameter set.

        Parameters
        ==========
        ax : AxesSubplot, optional
            An axes to draw on, otherwise one is created.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
           from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
           p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
           p.plot_all()

        """
        if ax is None:
            fig, ax = plt.subplots()

        ax = self.plot_principal_radii_of_gyration(ax=ax)
        ax = self.plot_principal_inertia_ellipsoids(ax=ax)
        ax = self.plot_geometry(ax=ax)
        ax = self.plot_mass_centers(ax=ax)

        return ax


class Meijaard2007WithFeedbackParameterSet(Meijaard2007ParameterSet):
    """Represents the parameters of the benchmark bicycle presented in
    [Meijaard2007]_ and with full state feedback control gains.

    The four bodies are:

    - B: rear frame + rigid rider
    - F: front wheel
    - H: front frame (fork & handlebars)
    - R: rear wheel

    Parameters
    ==========
    parameters : dictionary
        A dictionary mapping variable names to values that contains the
        following keys:

        - ``IBxx`` : x moment of inertia of the frame/rider [kg*m**2]
        - ``IBxz`` : xz product of inertia of the frame/rider [kg*m**2]
        - ``IBzz`` : z moment of inertia of the frame/rider [kg*m**2]
        - ``IFxx`` : x moment of inertia of the front wheel [kg*m**2]
        - ``IFyy`` : y moment of inertia of the front wheel [kg*m**2]
        - ``IHxx`` : x moment of inertia of the handlebar/fork [kg*m**2]
        - ``IHxz`` : xz product of inertia of the handlebar/fork [kg*m**2]
        - ``IHzz`` : z moment of inertia of the handlebar/fork [kg*m**2]
        - ``IRxx`` : x moment of inertia of the rear wheel [kg*m**2]
        - ``IRyy`` : y moment of inertia of the rear wheel [kg*m**2]
        - ``c`` : trail [m]
        - ``g`` : acceleration due to gravity [m/s**2]
        - ``kTdel_del`` : steer torque feedback gain from steer angle [N*m/rad]
        - ``kTdel_deld`` : steer torque feedback gain from steer rate [N*m*s/rad]
        - ``kTdel_phi`` : steer torque feedback gain from roll angle [N*m/rad]
        - ``kTdel_phid`` : steer torque feedback gain from roll rate [N*m*s/rad]
        - ``kTphi_del`` : roll torque feedback gain from steer angle [N*m/rad]
        - ``kTphi_deld`` : roll torque feedback gain from steer rate [N*m*s/rad]
        - ``kTphi_phi`` : roll torque feedback gain from roll angle [N*m/rad]
        - ``kTphi_phid`` : roll torque feedback gain from roll rate [N*m*s/rad]
        - ``lam`` : steer axis tilt [rad]
        - ``mB`` : frame/rider mass [kg]
        - ``mF`` : front wheel mass [kg]
        - ``mH`` : handlebar/fork assembly mass [kg]
        - ``mR`` : rear wheel mass [kg]
        - ``rF`` : front wheel radius [m]
        - ``rR`` : rear wheel radius [m]
        - ``w`` : wheelbase [m]
        - ``xB`` : x distance to the frame/rider center of mass [m]
        - ``xH`` : x distance to the frame/rider center of mass [m]
        - ``zB`` : z distance to the frame/rider center of mass [m]
        - ``zH`` : z distance to the frame/rider center of mass [m]

    includes_rider : boolean
        True if body B is the combined rear frame and rider in terms of
        mass and inertia values.

    Attributes
    ==========
    par_strings : dictionary
        Maps ASCII strings to their LaTeX string.
    body_labels : list of strings
        Single capital letters that correspond to the four rigid bodies in the
        model.

    References
    ==========

    .. [Meijaard2007] Meijaard J.P, Papadopoulos Jim M, Ruina Andy and Schwab
       A.L, 2007, Linearized dynamics equations for the balance and steer of a
       bicycle: a benchmark and review, Proc. R. Soc. A., 463:1955–1982
       http://doi.org/10.1098/rspa.2007.1857

    """

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
        'kTdel_del': r'k_{T_{\delta}\delta}',
        'kTdel_deld': r'k_{T_{\delta}\dot{\delta}}',
        'kTdel_phi': r'k_{T_{\delta}\phi}',
        'kTdel_phid': r'k_{T_{\delta}\dot{\phi}}',
        'kTphi_del': r'k_{T_{\phi}\delta}',
        'kTphi_deld': r'k_{T_{\phi}\dot{\delta}}',
        'kTphi_phi': r'k_{T_{\phi}\phi}',
        'kTphi_phid': r'k_{T_{\phi}\dot{\phi}}',
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
        'zH': r'z_H',
    }

    body_labels = ['B', 'F', 'H', 'R']

    def __init__(self, parameters, includes_rider):
        super().__init__(parameters, includes_rider)


class Moore2019ParameterSet(ParameterSet):
    """Represents the parameters of the bicycle parameterization presented in
    [1]_.

    The four bodies are:

    - D: rear frame
    - F: front wheel
    - H: front frame (fork & handlebars)
    - P: rigid rider
    - R: rear wheel

    Parameters
    ==========
    parameters : dictionary
        A dictionary mapping variable names to values.

    Attributes
    ==========
    par_strings : dictionary
        Maps ASCII strings to their LaTeX string.
    body_labels : list of strings
        Single capital letters that correspond to the five rigid bodies in the
        model.

    References
    ==========

    .. [1] Moore, Jason K.; Hubbard, Mont (2019): Expanded Optimization for
       Discovering Optimal Lateral Handling Bicycles. Proceedings of Bicycle
       and Motorcycle Dynamics 2019: A Symposium on the Dynamics and Control of
       Single Track Vehicles https://doi.org/10.6084/m9.figshare.9942938.v1

    """

    non_min_par_strings = {
        'alphaF': r'\alpha_F',
        'alphaR': r'\alpha_R',
        'kFbb': r'k_{Fbb}',
        'kRbb': r'k_{Rbb}',
        'yD': r'y_D',
        'yF': r'y_F',
        'yH': r'y-H',
        'yP': r'y_P',
        'yR': r'y_R',
        'zR': r'z_R',
        'zF': r'z_F',
        'xR': r'x_R',
        'xF': r'x_F',
    }
    # maps "Python" string to LaTeX version
    par_strings = {
        'alphaD': r'\alpha_D',
        'alphaH': r'\alpha_H',
        'alphaP': r'\alpha_P',
        'c': r'c',
        'g': r'g',
        'kDaa': r'k_{Daa}',
        'kDbb': r'k_{Dbb}',
        'kDyy': r'k_{Dyy}',
        'kFaa': r'k_{Faa}',
        'kFyy': r'k_{Fyy}',
        'kHaa': r'k_{Haa}',
        'kHbb': r'k_{Hbb}',
        'kHyy': r'k_{Hyy}',
        'kPaa': r'k_{Paa}',
        'kPbb': r'k_{Pbb}',
        'kPyy': r'k_{Pyy}',
        'kRaa': r'k_{Raa}',
        'kRyy': r'k_{Ryy}',
        'lP': r'l_P',
        'lam': r'\lambda',
        'mD': r'm_D',
        'mF': r'm_F',
        'mH': r'm_H',
        'mP': r'm_B',
        'mR': r'm_R',
        'rF': r'r_F',
        'rR': r'r_R',
        'v': r'v',
        'w': r'w',
        'wP': r'w_P',
        'xD': r'x_D',
        'xH': r'x_H',
        'xP': r'x_P',
        'zD': r'z_D',
        'zH': r'z-H',
        'zP': r'z_P',
    }

    body_labels = ['D', 'F', 'H', 'P', 'R']

    def __init__(self, parameters):
        super().__init__(parameters)
        self.parameters = parameters
        self._generate_body_colors()

    def _calc_derived_params(self):
        p = self.parameters

        pext = {}
        pext['alphaF'] = 0.0
        pext['alphaR'] = 0.0
        pext['yD'] = 0.0
        pext['yP'] = 0.0
        pext['yH'] = 0.0
        pext['yR'] = 0.0
        pext['yF'] = 0.0
        pext['xR'] = 0.0
        pext['xF'] = p['w']
        pext['zR'] = -p['rR']
        pext['zF'] = -p['rF']
        pext['kRbb'] = p['kRaa']
        pext['kFbb'] = p['kFaa']

        return pext

    def to_parameterization(self, name):
        """Returns a specific parameter set based on the provided
        parameterization name.

        Parameters
        ==========
        name : string
            The name of the parameterization. These should correspond to a
            subclass of a ``ParameterSet`` and the name will be the string that
            precedes "ParameterSet". For example, the parameterization name of
            ``Meijaard2007ParameterSet`` is ``Meijaard2007``.

        Returns
        =======
        ParmeterSet
            If a different parameterization is requested and this class can
            convert itself, it will return a new parameter set of the correct
            parameterization.

        Examples
        ========

        >>> from bicycleparameters.parameter_dicts import moore2019_browser_jason
        >>> from bicycleparameters.parameter_sets import Moore2019ParameterSet
        >>> moore_set = Moore2019ParameterSet(moore2019_browser_jason)
        >>> moore_set.mass_center_of('P', 'D')
        array([ 0.31156277,  0.        , -1.03972442])
        >>> meijaard_set = moore_set.to_parameterization('Meijaard2007')
        >>> meijaard_set.mass_center_of('B')
        array([ 0.31156277,  0.        , -1.03972442])

        """
        if name == self.parameterization:
            return self
        elif name == 'Meijaard2007':
            b = _convert_principal_to_benchmark(self.parameters)
            # Moore2019 always includes the rider
            return Meijaard2007ParameterSet(b, True)
        elif name == 'Meijaard2007WithFeedback':
            b = _convert_principal_to_benchmark(self.parameters)
            # Moore2019 always includes the rider
            par_set = Meijaard2007WithFeedbackParameterSet(b, True)
            return par_set.to_parameterization(name)
        else:
            msg = '{} is not an available parameter set'
            raise ValueError(msg.format(name))

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

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import moore2019_browser_jason
           from bicycleparameters.parameter_sets import Moore2019ParameterSet
           p = Moore2019ParameterSet(moore2019_browser_jason)
           p.plot_geometry()

        """

        p = self.parameters

        if ax is None:
            fig, ax = plt.subplots()

        ax = _plot_geometry_from_benchmark(ax, p,
                                           show_steer_axis=show_steer_axis)

        self._finalize_plot(ax)

        return ax

    def plot_person_diamond(self, show_cross=False, ax=None):
        """Plots a diamond that represents the approximate person's physical
        extents.

        Parameters
        ==========
        show_cross : boolean, optional
            Plots a cross in the diamond that spans opposite vertices.
        ax : AxesSubplot, optional
            An axes to draw on, otherwise one is created.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import moore2019_browser_jason
           from bicycleparameters.parameter_sets import Moore2019ParameterSet
           p = Moore2019ParameterSet(moore2019_browser_jason)
           p.plot_person_diamond()

        """

        if ax is None:
            fig, ax = plt.subplots()

        p = self.parameters

        x_offset = np.array([
            p['wP']/2*np.cos(-p['alphaP']),
            -p['lP']/2*np.cos(-p['alphaP'] + np.pi/2),
            -p['wP']/2*np.cos(-p['alphaP']),
            p['lP']/2*np.cos(-p['alphaP'] + np.pi/2),
            p['wP']/2*np.cos(-p['alphaP'])])
        x = p['xP'] + x_offset

        z_offset = np.array([
            p['wP']/2*np.sin(-p['alphaP']),
            -p['lP']/2*np.sin(-p['alphaP'] + np.pi/2),
            -p['wP']/2*np.sin(-p['alphaP']),
            p['lP']/2*np.sin(-p['alphaP'] + np.pi/2),
            p['wP']/2*np.sin(-p['alphaP'])])
        z = p['zP'] + z_offset

        ax.plot(x, z, color=self.body_colors['P'])

        if show_cross:
            ax.plot([p['xP'] - p['wP']/2*np.cos(-p['alphaP']),
                     p['xP'] + p['wP']/2*np.cos(-p['alphaP'])],
                    [p['zP'] - p['wP']/2*np.sin(-p['alphaP']),
                     p['zP'] + p['wP']/2*np.sin(-p['alphaP'])],
                    color='black', linewidth=2)

            ax.plot([p['xP'] - p['lP']/2*np.cos(-p['alphaP'] + np.pi/2),
                     p['xP'] + p['lP']/2*np.cos(-p['alphaP'] + np.pi/2)],
                    [p['zP'] - p['lP']/2*np.sin(-p['alphaP'] + np.pi/2),
                     p['zP'] + p['lP']/2*np.sin(-p['alphaP'] + np.pi/2)],
                    color='black', linewidth=2)

        self._finalize_plot(ax)

        return ax

    def plot_mass_centers(self, bodies=None, ax=None):
        """Returns a matplotlib axes with a mass center symbols for the
        specified bodies to the plot.

        Parameters
        ==========
        bodies: list of strings, optional
            A subset of the strings present in the class attribute
            ``body_labels``.
        ax: matplotlib Axes, optional
            An axes to plot on.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import moore2019_browser_jason
           from bicycleparameters.parameter_sets import Moore2019ParameterSet
           p = Moore2019ParameterSet(moore2019_browser_jason)
           p.plot_mass_centers()

        """

        if ax is None:
            fig, ax = plt.subplots()

        if bodies is None:
            bodies = self.body_labels

        for b in bodies:
            ax = self.plot_body_mass_center(b, ax=ax)

        self._finalize_plot(ax)

        return ax

    def plot_body_mass_center(self, body, ax=None):
        """Returns a matplotlib axes with a mass center symbol for the
        specified body to the plot.

        Parameters
        ==========
        body : string
            The body string: D, F, H, P, or R
        ax : SubplotAxes, optional
            Axes to plot on.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import moore2019_browser_jason
           from bicycleparameters.parameter_sets import Moore2019ParameterSet
           p = Moore2019ParameterSet(moore2019_browser_jason)
           p.plot_body_mass_center('D')

        """

        if ax is None:
            fig, ax = plt.subplots()

        p = self.parameters.copy()
        p.update(self._calc_derived_params())

        # mass center
        x = p['x{}'.format(body)]
        z = p['z{}'.format(body)]
        radius = max(p['w'], p['lP']) / 30
        ax = _com_symbol(ax, (x, z), radius,
                         color=self.body_colors[body], label=body)

        self._finalize_plot(ax)

        return ax

    def plot_principal_radii_of_gyration(self, bodies=None, ax=None):
        """Returns a matplotlib axes with lines and a circle that indicate the
        principal radii of gyration for all five bodies.

        Parameters
        ==========
        bodies : list of strings, optional
            Either ['D', 'F', 'H', 'P', 'R'] or a subset thereof.
        ax : AxesSubplot, optional
            An axes to draw on, otherwise one is created.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import moore2019_browser_jason
           from bicycleparameters.parameter_sets import Moore2019ParameterSet
           p = Moore2019ParameterSet(moore2019_browser_jason)
           p.plot_principal_radii_of_gyration()

        """

        if ax is None:
            fig, ax = plt.subplots()

        if bodies is None:
            bodies = ['D', 'F', 'H', 'P', 'R']

        for b in bodies:
            ax = self.plot_body_principal_radii_of_gyration(b, ax=ax)

        self._finalize_plot(ax)

        return ax

    def plot_body_principal_radii_of_gyration(self, body, ax=None):
        """Returns a matplotlib axes with lines and a circle that indicate the
        principal radii of gyration of the specified body.

        Parameters
        ==========
        body : string
            The body string: D, F, H, P, or R
        ax : SubplotAxes, optional
            Axes to plot on.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import moore2019_browser_jason
           from bicycleparameters.parameter_sets import Moore2019ParameterSet
           p = Moore2019ParameterSet(moore2019_browser_jason)
           p.plot_body_principal_radii_of_gyration('P')

        """

        if ax is None:
            fig, ax = plt.subplots()

        p = self.parameters.copy()
        p.update(self._calc_derived_params())

        x = p['x{}'.format(body)]
        z = p['z{}'.format(body)]
        kyy = p['k{}yy'.format(body)]
        kaa = p['k{}aa'.format(body)]
        kbb = p['k{}bb'.format(body)]
        alpha = p['alpha{}'.format(body)]  # angle between x and aa about y

        linestyle = '--'

        c = patches.Circle((x, z), radius=kyy, fill=False,
                           color=self.body_colors[body], linestyle=linestyle)
        ax.add_patch(c)

        # NOTE : -alpha is required because we are mapping the xz axes to a new
        # planar drawing grid which is x and y with z pointing out of the
        # screen
        ax.plot([x - kbb*np.cos(-alpha), x + kbb*np.cos(-alpha)],
                [z - kbb*np.sin(-alpha), z + kbb*np.sin(-alpha)],
                color=self.body_colors[body], linestyle=linestyle)

        ax.plot([x - kaa*np.cos(-alpha - np.pi/2),
                 x + kaa*np.cos(-alpha - np.pi/2)],
                [z - kaa*np.sin(-alpha - np.pi/2),
                 z + kaa*np.sin(-alpha - np.pi/2)],
                color=self.body_colors[body], linestyle=linestyle)

        self._finalize_plot(ax)

        return ax

    def plot_body_principal_inertia_ellipsoid(self, body, ax=None):
        """Returns a matplotlib axes with an ellipse that respresnts the XZ
        plane view of a constant density ellipsoid which has the same principal
        moments and axes of inertia as the body.

        Parameters
        ==========
        body : string
            The body string: D, F, H, P, or R
        ax : SubplotAxes, optional
            Axes to plot on.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import moore2019_browser_jason
           from bicycleparameters.parameter_sets import Moore2019ParameterSet
           p = Moore2019ParameterSet(moore2019_browser_jason)
           p.plot_body_principal_inertia_ellipsoid('P')

        """

        if ax is None:
            fig, ax = plt.subplots()

        p = self.parameters.copy()
        p.update(self._calc_derived_params())

        kaa = p['k{}aa'.format(body)]
        kbb = p['k{}bb'.format(body)]
        kyy = p['k{}yy'.format(body)]
        alpha = p['alpha{}'.format(body)]

        width = np.sqrt(5/2*(-kaa**2 + kyy**2 + kbb**2))
        height = np.sqrt(5/2*(kaa**2 + kyy**2 - kbb**2))

        ellipse = patches.Ellipse((p['x{}'.format(body)],
                                   p['z{}'.format(body)]), width, height,
                                  angle=-np.rad2deg(alpha), fill=False,
                                  color=self.body_colors[body])
        ax.add_patch(ellipse)

        self._finalize_plot(ax)

        return ax

    def plot_principal_inertia_ellipsoids(self, bodies=None, ax=None):
        """Returns a Matplotlib axes with 2D representations of 3D solid
        uniform ellipsoids that have the same inertia as the body.

        Parameters
        ==========
        bodies: list of strings, optional
            A subset of the strings present in the class attribute
            ``body_labels``.
        ax: matplotlib Axes, optional
            An axes to plot on.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import moore2019_browser_jason
           from bicycleparameters.parameter_sets import Moore2019ParameterSet
           p = Moore2019ParameterSet(moore2019_browser_jason)
           p.plot_principal_inertia_ellipsoids()

        """
        if ax is None:
            fig, ax = plt.subplots()

        if bodies is None:
            bodies = self.body_labels

        for b in bodies:
            ax = self.plot_body_principal_inertia_ellipsoid(b, ax=ax)

        self._finalize_plot(ax)

        return ax

    def plot_all(self, ax=None):
        """Returns matplotlib axes with the geometry and inertial
        representations of all bodies of the bicycle parameter set.

        Parameters
        ==========
        ax: matplotlib Axes, optional
            An axes to plot on.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           from bicycleparameters.parameter_dicts import moore2019_browser_jason
           from bicycleparameters.parameter_sets import Moore2019ParameterSet
           p = Moore2019ParameterSet(moore2019_browser_jason)
           p.plot_all()

        """

        if ax is None:
            fig, ax = plt.subplots()

        ax = self.plot_principal_radii_of_gyration(ax=ax)
        ax = self.plot_principal_inertia_ellipsoids(ax=ax)
        ax = self.plot_person_diamond(ax=ax)
        ax = self.plot_geometry(ax=ax)
        ax = self.plot_mass_centers(ax=ax)

        self._finalize_plot(ax)

        return ax

    def mass_center_of(self, *bodies):
        """Returns the vector locating the center of mass of the collection of
        bodies.

        Parameters
        ==========
        bodies : iterable of strings
            Subset from ``body_labels``.

        Returns
        =======
        com : ndarray, shape(3,)
            Vector locating the center of mass of the bodies given in
            ``bodies``.

        """
        if len(bodies) == 1:
            return self.form_mass_center_vector(bodies[0])
        else:
            coordinates = []
            masses = []
            for body in bodies:
                masses.append(self.parameters['m{}'.format(body)])
                coordinates.append(
                    self.form_mass_center_vector(body))

            coordinates = np.array(coordinates).T

            mass, com = total_com(coordinates, masses)

            return com

    def form_mass_center_vector(self, body):
        """Returns an array representing the vector to the mass center of the
        body.

        Parameters
        ==========
        body : string
            One of 'P', 'D', 'F', 'H', 'R'.

        Returns
        =======
        ndarray, shape(3,)
            A vector containing the X, Y, and X coordinates of the mass center
            of the body.

        """

        p = self.parameters.copy()
        p.update(self._calc_derived_params())

        x = p['x{}'.format(body)]
        y = p['y{}'.format(body)]
        z = p['z{}'.format(body)]

        return np.array([x, y, z])


class Moore2012ParameterSet(ParameterSet):
    """Represents the parameters of the Carvallo-Whipple model presented in
    [Moore2012]_.

    The four bodies are:

    - C: rear frame + rigid rider
    - D: rear wheel
    - E: front frame (fork & handlebars)
    - F: front wheel

    Parameters
    ==========
    parameters : dictionary
        A dictionary mapping variable names to values that contains the
        following keys:

        - 'd1'
        - 'd2'
        - 'd3'
        - 'g'
        - 'ic11'
        - 'ic22'
        - 'ic31'
        - 'ic33'
        - 'id11'
        - 'id22'
        - 'ie11'
        - 'ie22'
        - 'ie31'
        - 'ie33'
        - 'if11'
        - 'if22'
        - 'l1'
        - 'l2'
        - 'l3'
        - 'l4'
        - 'mc'
        - 'md'
        - 'me'
        - 'mf'
        - 'rf'
        - 'rr'
        - 'v'

    includes_rider : boolean
        True if body C is the combined rear frame and rider in terms of
        mass and inertia values.

    Attributes
    ==========
    par_strings : dictionary
        Maps ASCII strings to their LaTeX string.
    body_labels : list of strings
        Single letters that correspond to the four rigid bodies in the model.

    """

    par_strings = {
        'd1': r'd_1',
        'd2': r'd_2',
        'd3': r'd_3',
        'g': r'g',
        'ic11': r'I_{C11}',
        'ic22': r'I_{C22}',
        'ic31': r'I_{C31}',
        'ic33': r'I_{C33}',
        'id11': r'I_{D11}',
        'id22': r'I_{D22}',
        'ie11': r'I_{D11}',
        'ie22': r'I_{E22}',
        'ie31': r'I_{E31}',
        'ie33': r'I_{E33}',
        'if11': r'I_{F11}',
        'if22': r'I_{F22}',
        'l1': r'l_1',
        'l2': r'l_2',
        'l3': r'l_3',
        'l4': r'l_4',
        'mc': r'm_C',
        'md': r'm_D',
        'me': r'm_E',
        'mf': r'm_F',
        'rf': r'r_f',
        'rr': r'r_r',
        'v': 'v',
    }

    body_labels = ['C', 'D', 'E', 'F']

    def __init__(self, parameters, includes_rider):
        super().__init__(parameters)
        self.parameters = parameters
        self.includes_rider = includes_rider
        self._generate_body_colors()


class MooreRiderLean2012ParameterSet(ParameterSet):
    """Represents the parameters of the Carvallo-Whipple model with a leaning
    rider presented in [Moore2012]_.

    The four bodies are:

    - C: rear frame + rigid rider
    - D: rear wheel
    - E: front frame (fork & handlebars)
    - F: front wheel

    Parameters
    ==========
    parameters : dictionary
        A dictionary mapping variable names to values that contains the
        following keys:

       - 'c9'
       - 'd1'
       - 'd2'
       - 'd3'
       - 'd4'
       - 'g'
       - 'ic11'
       - 'ic22'
       - 'ic31'
       - 'ic33'
       - 'id11'
       - 'id22'
       - 'ie11'
       - 'ie22'
       - 'ie31'
       - 'ie33'
       - 'if11'
       - 'if22'
       - 'ig11'
       - 'ig22'
       - 'ig31'
       - 'ig33'
       - 'k9'
       - 'l1'
       - 'l2'
       - 'l3'
       - 'l4'
       - 'l5'
       - 'l6'
       - 'mc'
       - 'md'
       - 'me'
       - 'mf'
       - 'mg'
       - 'rf'
       - 'rr'
       - 'v'

    Attributes
    ==========
    par_strings : dictionary
        Maps ASCII strings to their LaTeX string.
    body_labels : list of strings
        Single letters that correspond to the four rigid bodies in the model.

    """
    par_strings = {
        'c9': r'c_9',
        'd1': r'd_1',
        'd2': r'd_2',
        'd3': r'd_3',
        'd4': r'd_4',
        'g': r'g',
        'ic11': r'I_{C11}',
        'ic22': r'I_{C22}',
        'ic31': r'I_{C31}',
        'ic33': r'I_{C33}',
        'id11': r'I_{D11}',
        'id22': r'I_{D22}',
        'ie11': r'I_{D11}',
        'ie22': r'I_{E22}',
        'ie31': r'I_{E31}',
        'ie33': r'I_{E33}',
        'if11': r'I_{F11}',
        'if22': r'I_{F22}',
        'ig11': r'I_{G11}',
        'ig22': r'I_{G22}',
        'ig31': r'I_{G31}',
        'ig33': r'I_{G33}',
        'k9': r'k_9',
        'l1': r'l_1',
        'l2': r'l_2',
        'l3': r'l_3',
        'l4': r'l_4',
        'l5': r'l_5',
        'l6': r'l_6',
        'mc': r'm_C',
        'md': r'm_D',
        'me': r'm_E',
        'mf': r'm_F',
        'mg': r'm_G',
        'rf': r'r_f',
        'rr': r'r_r',
        'v': 'v',
    }

    body_labels = ['C', 'D', 'E', 'F', 'G']

    def __init__(self, parameters):
        super().__init__(parameters)
        self.parameters = parameters
        self._generate_body_colors()
