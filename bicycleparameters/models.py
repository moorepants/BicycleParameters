import abc
import itertools
import warnings

import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
from dtk.bicycle import pitch_from_roll_and_steer

from .bicycle import benchmark_par_to_canonical, ab_matrix, sort_eigenmodes


class _Model(abc.ABC):
    """A model is a set of differential algebraic equations in time that have:
    constants and time varying (coordinates, speeds, and exogenous inputs).
    The model can be nonlinear, linear, have algebraic constraints, or not.  A
    parameter set is associated with a particular parameterization of one or
    more models.

    """
    input_vars = ['r1', 'r2', 'r3']
    state_vars = ['q1', 'q2', 'u1', 'u2']
    input_units = ['N', 'N', 'N-m']
    state_units = ['m', 'rad', 'm/s', 'rad/s']
    input_vars_latex = ['r_1', 'r_2', 'r_3']
    state_vars_latex = ['q_1', 'q_2', 'u_1', 'u_2']

    def _parse_parameter_overrides(self, **parameter_overrides):
        """Returns the model's parameter dictionary with the overridden
        parameters replaced.

        Parameters
        ==========
        parameter_overrides : dictionary
            Parameter keys that map to floats or array_like of floats
            shape(n,). All keys that map to array_like must be of the same
            length.

        Returns
        =======
        par : dictionary
            Copy of self.parameter_set.parameters with overridden parameter
            values.
        array_keys : list
            All parameter key strings that hold arrays.
        array_len : None or int
            If there are arrays, this is the common length.

        """

        par = self.parameter_set.parameters.copy()

        array_len = None
        array_keys = []

        for key, val in parameter_overrides.items():
            if key not in par.keys():  # don't add invalid keys
                msg = '{} is not a valid parameter, ignoring'
                warnings.warn(msg.format(key))
            else:
                if np.isscalar(val):
                    par[key] = float(val)
                else:  # is an array
                    if array_len is None:
                        array_len = len(val)
                    if len(val) != array_len:
                        msg = ('All array valued parameters must have the '
                               'same length.')
                        raise ValueError(msg)
                    array_keys.append(key)
                    par[key] = val

        return par, array_keys, array_len

    @abc.abstractmethod
    def form_state_space_matrices(self, **parameter_overrides):
        A = np.eye(4)
        B = np.zeros((4, 2))
        return A, B

    def calc_eigen(self, left=False, **parameter_overrides):
        """Returns the right (or left) eigenvalues and eigenvectors of the
        linear model.

        Parameters
        ==========
        left : boolean, optional
            If true, the left eigenvectors will be returned, i.e.
            ``A.T*v=lam*v``.
        **parameter_overrides : dictionary
            Parameter keys that map to floats or array_like of floats
            shape(n,). All keys that map to array_like must be of the same
            length.

        Returns
        =======
        evals : ndarray, shape(4,) or shape (n,4)
            Eigenvalues.
        evecs : ndarray, shape(4,4) or shape (n,4,4)
            Eigenvectors, each columns are eigenvectors and are associated with
            same index of the eigenvalues.

        Examples
        ========

        >>> from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
        >>> from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
        >>> from bicycleparameters.models import Meijaard2007Model
        >>> p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
        >>> m = Meijaard2007Model(p)
        >>> evals, evecs = m.calc_eigen()
        >>> evals
        array([-6.74423162+0.j        , -2.9146438 +0.j        ,
                3.39244999+0.61085077j,  3.39244999-0.61085077j])
        >>> evecs
        array([[ 0.00197344+0.j        , -0.2953538 +0.j        ,
                 0.04320146-0.0753826j ,  0.04320146+0.0753826j ],
               [ 0.14665803+0.j        ,  0.13447333+0.j        ,
                -0.26053575+0.04691255j, -0.26053575-0.04691255j],
               [-0.01330934+0.j        ,  0.86085111+0.j        ,
                 0.1926063 -0.22934205j,  0.1926063 +0.22934205j],
               [-0.98909574+0.j        , -0.39194186+0.j        ,
                -0.91251108+0.j        , -0.91251108-0.j        ]])

        """
        A, B = self.form_state_space_matrices(**parameter_overrides)

        if len(A.shape) == 3:  # array version
            evals = np.zeros(A.shape[:2], dtype='complex128')
            evecs = np.zeros(A.shape, dtype='complex128')
            for i, Ai in enumerate(A):
                if left:
                    Ai = np.transpose(Ai)
                evals[i], evecs[i] = np.linalg.eig(Ai)
            return evals, evecs
        else:
            if left:
                A = np.transpose(A)
            return np.linalg.eig(A)

    def plot_eigenvalue_parts(self, ax=None, colors=None,
                              show_stable_regions=True, hide_zeros=False,
                              sort_modes=True,
                              **parameter_overrides):
        """Returns a matplotlib axis of the real and imaginary parts of the
        eigenvalues plotted against the provided parameter.

        Parameters
        ==========
        ax : Axes
            Matplotlib axes.
        colors : sequence, len(4)
            Matplotlib colors for the 4 modes.
        show_stable_regions : boolean, optional
            If true, a grey shaded background will indicate stable regions.
        sort_modes : boolean, optional
            Sorting the modes does not always work so you can disable it.
        hide_zeros : boolean or float, optional
            If true, real or imaginary parts that are smaller than 1e-12 will
            not be plotted. Providing a float will set the tolerance.
        **parameter_overrides : dictionary
            Parameter keys that map to floats or array_like of floats
            shape(n,). All keys that map to array_like must be of the same
            length.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           import numpy as np
           from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
           from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
           from bicycleparameters.models import Meijaard2007Model
           p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
           m = Meijaard2007Model(p)
           m.plot_eigenvalue_parts(v=np.linspace(0.0, 10.0, num=101))

        """

        if ax is None:
            fig, ax = plt.subplots()

        evals, evecs = self.calc_eigen(**parameter_overrides)
        if len(evals.shape) > 1:
            if sort_modes:
                evals, evecs = sort_eigenmodes(evals, evecs)
            legend = ['M{}'.format(i + 1) for i in range(evals.shape[1])]*2
        else:
            evals, evecs = np.array([evals]), np.array([evecs])
            # TODO : legend if not versus param

        tol = hide_zeros if isinstance(hide_zeros, float) else 1e-14

        par, array_keys, _ = self._parse_parameter_overrides(
            **parameter_overrides)

        if show_stable_regions:
            ax.fill_between(par[array_keys[0]],
                            np.min([np.min(evals.real), np.min(evals.imag)]),
                            np.max([np.max(evals.real), np.max(evals.imag)]),
                            # NOTE : include zero here for A matrices that
                            # produce zero eigenvalues.
                            where=np.all(evals.real <= 0.0, axis=1),
                            color='grey',
                            alpha=0.25,
                            transform=ax.get_xaxis_transform())

        if colors is None:
            colors_was_none = True
            colors = itertools.cycle(mplcolors.TABLEAU_COLORS)
        else:
            colors_was_none = False

        # imaginary components
        for eval_sequence, color, label in zip(evals.T, colors, legend):
            imag_vals = np.abs(np.imag(eval_sequence))
            if hide_zeros:
                imag_vals[np.abs(imag_vals) < tol] = np.nan
            if sort_modes:
                line = {'linestyle': '--', 'color': color}
            else:
                line = {'linestyle': 'none', 'marker': '.', 'markersize': 1,
                        'color': 'C1'}
            ax.plot(par[array_keys[0]], imag_vals, label=label, **line)

        if sort_modes and colors_was_none:
            # reset the cycle
            colors = itertools.cycle(mplcolors.TABLEAU_COLORS)

        # plot the real parts of the eigenvalues
        for eval_sequence, color, label in zip(evals.T, colors, legend):
            real_vals = np.real(eval_sequence)
            if hide_zeros:
                real_vals[np.abs(real_vals) < tol] = np.nan
            if sort_modes:
                line = {'linestyle': '-', 'color': color}
            else:
                line = {'linestyle': 'none', 'marker': '.', 'markersize': 1,
                        'color': 'black'}
            ax.plot(par[array_keys[0]], real_vals, label=label, **line)

        # set labels and limits
        ax.set_ylabel('Real and Imaginary Parts of the Eigenvalue [1/s]')
        ax.set_xlim((par[array_keys[0]][0], par[array_keys[0]][-1]))

        ax.grid()

        if sort_modes:
            ax.legend(ncol=5)

        ax.set_xlabel(
            '$' + self.parameter_set.par_strings[array_keys[0]] + '$')

        return ax

    def plot_eigenvectors(self, hide_zeros=False, **parameter_overrides):
        """Plots the components of the eigenvectors in the real and imaginary
        plane.

        Parameters
        ==========
        **parameter_overrides : dictionary
            Parameter keys that map to floats or array_like of floats
            shape(n,). All keys that map to array_like must be of the same
            length.

        Returns
        =======
        axes : ndarray, shape(n, 4)
            Polar plot axes for each eigenvector (columns). The rows correspond
            to a varied parameter.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           import numpy as np
           from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
           from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
           from bicycleparameters.models import Meijaard2007Model
           p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
           m = Meijaard2007Model(p)
           m.plot_eigenvectors(v=[1.0, 3.0, 5.0])

        """
        par, arr_keys, _ = self._parse_parameter_overrides(
            **parameter_overrides)

        eval_seq, evec_seq = self.calc_eigen(**parameter_overrides)
        eval_seq, evec_seq = np.atleast_2d(eval_seq), np.atleast_3d(evec_seq)

        # TODO : This assumes that the eigenvalues are in the same order.
        plot = np.ones(len(eval_seq[-1]), dtype=int)
        if hide_zeros:
            tol = hide_zeros if isinstance(hide_zeros, float) else 1e-14
            for i, ev in enumerate(eval_seq[-1]):
                if np.abs(ev) < tol:
                    plot[i] = 0

        if eval_seq.shape[0] > 10:
            msg = ('Plots will be too large, use fewer than 11 values in the '
                   'varied parameter.')
            raise ValueError(msg)

        # TODO : sort_eigenmodes() is doing something funny and not adding the
        # 4th eigenvalue, so you often end up with duplicates eigenvalues for
        # one eigenval and one missing. Also the algorithm may not work well
        # with spaced out eigenvalues, which is what I've been trying here. You
        # may have to calculate eigenvals/vec across closer spacing, then
        # sample out the ones you want. For now, we don't sort coarse spaced
        # eigenvalues.
        # if arr_keys:
            # eval_seq, evec_seq = sort_eigenmodes(eval_seq, evec_seq)

        fig, axes = plt.subplots(eval_seq.shape[0],
                                 np.count_nonzero(plot),
                                 subplot_kw={'projection': 'polar'},
                                 layout='constrained')
        axes = np.atleast_2d(axes)
        fig.set_size_inches(axes.shape[1]*3, axes.shape[0]*3)
        lw = list(range(1, len(self.state_vars) + 1))
        lw.reverse()

        if arr_keys:
            par_vals = par[arr_keys[0]]
        else:
            par_vals = [0.0]

        for k, (evals, evecs, par_val) in enumerate(zip(eval_seq, evec_seq.T,
                                                        par_vals)):

            if arr_keys:
                axes[k, 0].set_ylabel('{} = {:1.2f}'.format(arr_keys[0],
                                                            par_val),
                                      labelpad=30)

            i = 0
            for eigenval, eigenvec, pl in zip(evals, evecs, plot):

                if pl:
                    max_com = np.abs(eigenvec).max()

                    for j, component in enumerate(eigenvec):

                        radius = np.abs(component)/max_com
                        theta = np.angle(component)
                        axes[k, i].plot([0, theta], [0, radius], lw=lw[j])

                    axes[k, i].set_rmax(1.0)
                    msg = r'Eigenvalue: {:1.3f}'
                    if eigenval.real >= 0.0:
                        fontcolor = 'red'  # red indicates unstable
                    else:
                        fontcolor = 'black'
                    axes[k, i].set_title(msg.format(eigenval),
                                         fontdict={'color': fontcolor})
                    i += 1

        axes[0, 0].legend(['$' + s + '$' for s in self.state_vars_latex],
                          loc='upper center', bbox_to_anchor=(0.5, 1.05),
                          fancybox=True, shadow=True, ncol=4)

        return axes

    def simulate(self, times, initial_conditions, input_func=None,
                 **parameter_overrides):
        """Returns the state and input trajectories at each time value.

        Parameters
        ==========
        times : array_like, shape(n,)
            Monotonic increasing time values to simulate over.
        initial_conditions : array_like, shape(4,)
            Initial values of the states.
        input_func : function
            Takes form u = f(t, x) where u is array_like, shape(2,).
        **parameter_overrides : dictionary
            Parameter keys that map to floats or array_like of floats
            shape(n,). All keys that map to array_like must be of the same
            length.

        Returns
        =======
        states : ndarray, shape(n, 4)
            State trajectories over n time values.
        inputs : ndatrray, shape(n, 2)
            Input trajectories over n time values.

        """

        par, arr_keys, _ = self._parse_parameter_overrides(
            **parameter_overrides)

        if arr_keys:
            raise ValueError('Can only simulate with fixed parameters.')

        A, B = self.form_state_space_matrices(**parameter_overrides)

        if input_func is None:
            def eval_rhs(t, x):
                return A@x
        else:
            def eval_rhs(t, x):
                return A@x + B@input_func(t, x)

        res = spi.solve_ivp(eval_rhs,
                            (times[0], times[-1]),
                            initial_conditions,
                            t_eval=times,
                            method="LSODA")

        if input_func is None:
            inputs = np.zeros((len(times), B.shape[1]))
        else:
            inputs = np.empty((len(times), B.shape[1]))
            for i, ti in enumerate(times):
                ui = input_func(ti, res.y[:, i])
                inputs[i, :] = ui[:]

        return res.y.T, inputs

    def simulate_modes(self, times, **parameter_overrides):
        """Returns simulation results showing the behavior of each
        eigenmode.

        Parameters
        ==========
        times : array_like, shape(n,)
            Monotonic increasing time values to simulate over.
        **parameter_overrides : dictionary
            Parameter keys that map to floats or array_like of floats
            shape(n,). All keys that map to array_like must be of the same
            length.

        Returns
        =======
        results : ndarray, shape(4, n, 4)
            State trajectories for each mode with the shape corresponding to
            (mode, time, state).

        """

        par, arr_keys, _ = self._parse_parameter_overrides(
            **parameter_overrides)

        if arr_keys:
            raise ValueError('Can only simulate with fixed parameters.')

        A, B = self.form_state_space_matrices(**parameter_overrides)
        evals, evecs = self.calc_eigen(**parameter_overrides)

        def eval_rhs(t, x):
            return A@x

        results = np.empty((len(evals), len(times), len(evals)))

        for i, evec in enumerate(evecs.T):
            initial_condition = evec.real

            sim_res = spi.solve_ivp(eval_rhs, (times[0], times[-1]),
                                    initial_condition, t_eval=times)
            results[i] = sim_res.y.T

        return results

    def plot_mode_simulations(self, times, hide_zeros=False,
                              **parameter_overrides):
        """Returns matplotlib subplot axes with a simulation of each mode.

        Parameters
        ==========
        times : array_like, shape(n,)
            Monotonic increasing time values to simulate over.
        **parameter_overrides : dictionary
            Parameter keys that map to floats or array_like of floats
            shape(n,). All keys that map to array_like must be of the same
            length.

        Returns
        =======
        axes : ndarray, shape(4,2)
            Subplot axes with the modes on the rows and the angles in the first
            column and the angular rates in the second column.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           import numpy as np
           from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
           from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
           from bicycleparameters.models import Meijaard2007Model
           p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
           m = Meijaard2007Model(p)
           times = np.linspace(0.0, 5.0, num=51)
           m.plot_mode_simulations(times, v=6.0)

        """
        results = self.simulate_modes(times, **parameter_overrides)
        evals, evecs = self.calc_eigen(**parameter_overrides)

        unique_units = sorted(list(set(self.state_units)))

        plot = np.ones_like(evals)
        if hide_zeros:
            tol = hide_zeros if isinstance(hide_zeros, float) else 1e-12
            for i, ev in enumerate(evals):
                if np.abs(ev) < tol:
                    plot[i] = 0

        fig, axes = plt.subplots(np.count_nonzero(plot),
                                 len(unique_units), sharex=True,
                                 figsize=(len(unique_units)*3.0,
                                          np.count_nonzero(plot)*2.0),
                                 layout='constrained')

        i = 0
        for res, e_val, pl in zip(results, evals, plot):
            if pl:
                for state_traj, unit, name in zip(res.T, self.state_units,
                                                  self.state_vars_latex):
                    if 'rad' in unit:
                        state_traj = np.rad2deg(state_traj)
                    axes[i, unique_units.index(unit)].plot(
                        times, state_traj, label='$' + name + '$')


                msg = r'Eigenvalue: {:1.3f}'
                if e_val.real >= 0.0:
                    fontcolor = 'red'  # red indicates unstable
                else:
                    fontcolor = 'black'
                for j, unit in enumerate(unique_units):
                    axes[i, j].set_ylabel('[{}]'.format(unit))
                    axes[i, j].set_title(msg.format(e_val),
                                         fontdict={'color': fontcolor})
                    axes[i, j].legend()
                i += 1

        axes[-1, 0].set_xlabel('Time [s]')
        axes[-1, 1].set_xlabel('Time [s]')

        return axes


class Meijaard2007Model(_Model):
    """Carvallo-Whipple model presented in [Meijaard2007]_. It is both linear
    and the minimal model in terms of states and coordinates that fully
    describe the vehicle's dynamics: self-stability and non-minimum phase
    behavior.

    Parameters
    ==========
    parameter_set : ParameterSet
        The ``paramter_set.to_parameterization('meijaard2007')`` must
        return a dictionary that maps floats to the parameter keys
        containing:

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
        - ``v`` : speed [m/s]
        - ``w`` : wheelbase [m]
        - ``xB`` : x distance to the frame/rider center of mass [m]
        - ``xH`` : x distance to the frame/rider center of mass [m]
        - ``zB`` : z distance to the frame/rider center of mass [m]
        - ``zH`` : z distance to the frame/rider center of mass [m]

    Attributes
    ==========
    input_vars : list of strings
        Ordered list of ASCII strings that name the model's input variables.
    state_vars : list of strings
        Ordered list of ASCII strings that name the model's state variables.
    input_vars_latex : list of raw strings
        Ordered list of LaTeX strings that name the model's input variables.
    state_vars_latex : list of raw strings
        Ordered list of LaTeX strings that name the model's state variables.

    References
    ==========

    .. [Meijaard2007] Meijaard J.P, Papadopoulos Jim M, Ruina Andy and Schwab
       A.L, 2007, Linearized dynamics equations for the balance and steer of a
       bicycle: a benchmark and review, Proc. R. Soc. A., 463:1955–1982
       http://doi.org/10.1098/rspa.2007.1857

    """
    input_vars = ['Tphi', 'Tdelta']
    state_vars = ['phi', 'delta', 'phidot', 'deltadot']
    input_units = ['N-m', 'N-m']
    state_units = ['rad', 'rad', 'rad/s', 'rad/s']
    input_vars_latex = [r'T_\phi', r'T_\delta']
    state_vars_latex = [r'\phi', r'\delta', r'\dot{\phi}', r'\dot{\delta}']

    def __init__(self, parameter_set):
        self.parameter_set = parameter_set.to_parameterization('Meijaard2007')

    def form_reduced_canonical_matrices(self, **parameter_overrides):
        """Returns the canonical speed and gravity independent matrices for the
        Whipple-Carvallo bicycle model linearized about the nominal upright
        configuration.

        Parameters
        ==========
        **parameter_overrides : dictionary
            Parameter keys that map to floats or array_like of floats
            shape(n,). All keys that map to array_like must be of the same
            length.

        Returns
        =======
        M : ndarray, shape(2,2) or shape(n,2,2)
            Mass matrix.
        C1 : ndarray, shape(2,2) or shape(n,2,2)
            Velocity independent damping matrix.
        K0 : ndarray, shape(2,2) or shape(n,2,2)
            Gravity independent part of the stiffness matrix.
        K2 : ndarray, shape(2,2) or shape(n,2,2)
            Velocity squared independent part of the stiffness matrix.

        Notes
        =====
        The canonical matrices complete the following equation:

        ``M*q'' + v*C1*q' + [g*K0 + v**2*K2]*q = f``

        where:

        - ``q = [phi, delta]``
        - ``f = [Tphi, Tdelta]``

        ``phi``
            Bicycle roll angle.
        ``delta``
            Steer angle.
        ``Tphi``
            Roll torque.
        ``Tdelta``
            Steer torque.
        ``v``
            Bicylce longitudinal speed.
        ``g``
            Acceleration due to gravity.

        Examples
        ========

        >>> from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
        >>> from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
        >>> from bicycleparameters.models import Meijaard2007Model
        >>> p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
        >>> m = Meijaard2007Model(p)
        >>> M, C1, K0, K2 = m.form_reduced_canonical_matrices()
        >>> M
        array([[102.78013216,   1.53582801],
               [  1.53582801,   0.24890226]])
        >>> C1
        array([[ 0.       , 26.3947333],
               [-0.4503006,  1.037066 ]])
        >>> K0
        array([[-89.32195981,  -1.74159477],
               [ -1.74159477,  -0.67769624]])
        >>> K2
        array([[ 0.        , 74.12543   ],
               [ 0.        ,  1.57021553]])
        >>> M, _, _, _ = m.form_reduced_canonical_matrices(mB=150.0)
        >>> M
        array([[176.52178763,   2.69074048],
               [  2.69074048,   0.26699004]])

        """
        par, array_keys, array_len = self._parse_parameter_overrides(
            **parameter_overrides)

        mutable_par = par.copy()

        # these are the only variables needed to calculate M, C1, K0, K2
        canon_deps = ['IBxx', 'IBxz', 'IByy', 'IBzz',
                      'IFxx', 'IFyy',
                      'IHxx', 'IHxz', 'IHyy', 'IHzz',
                      'IRxx', 'IRyy',
                      'c', 'lam',
                      'mB', 'mF', 'mH', 'mR',
                      'rF', 'rR', 'w',
                      'xB', 'xH', 'zB', 'zH']

        compute_arrays = False
        for key in array_keys:
            if key in canon_deps:
                compute_arrays = True
                break

        if array_keys and compute_arrays:
            n = array_len
            M = np.zeros((n, 2, 2))
            C1 = np.zeros((n, 2, 2))
            K0 = np.zeros((n, 2, 2))
            K2 = np.zeros((n, 2, 2))
            for i in range(n):
                for key in array_keys:
                    mutable_par[key] = par[key][i]
                M[i], C1[i], K0[i], K2[i] = benchmark_par_to_canonical(
                    mutable_par)
            return M, C1, K0, K2
        else:
            return benchmark_par_to_canonical(par)

    def form_state_space_matrices(self, **parameter_overrides):
        """Returns the A and B matrices for the Whipple-Carvallo model
        linearized about the upright constant velocity configuration.

        Parameters
        ==========
        **parameter_overrides : dictionary
            Parameter keys that map to floats or array_like of floats
            shape(n,). All keys that map to array_like must be of the same
            length.

        Returns
        =======
        A : ndarray, shape(4,4) or shape(n,4,4)
            The state matrix.
        B : ndarray, shape(4,2) or shape(n,4,2)
            The input matrix.

        Notes
        =====
        ``A`` and ``B`` describe the Whipple model in state space form:

        ``x' = A * x + B * u``

        where the states are::

            x = |roll angle | = |phi     |
                |steer angle|   |delta   |
                |roll rate  |   |phidot  |
                |steer rate |   |deltadot|

        and the inputs are::

            u = |roll torque | = |Tphi  |
                |steer torque|   |Tdelta|

        Examples
        ========

        >>> from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
        >>> from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
        >>> from bicycleparameters.models import Meijaard2007Model
        >>> p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
        >>> m = Meijaard2007Model(p)
        >>> A, B = m.form_state_space_matrices()
        >>> A
        array([[ 0.        ,  0.        ,  1.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  1.        ],
               [ 8.26150335, -0.9471634 , -0.02977958, -0.21430735],
               [17.66475151, 26.24590352,  1.99289841, -2.84419587]])
        >>> B
        array([[ 0.        ,  0.        ],
               [ 0.        ,  0.        ],
               [ 0.01071772, -0.06613267],
               [-0.06613267,  4.42570676]])

        """
        par, array_keys, array_len = self._parse_parameter_overrides(
            **parameter_overrides)

        M, C1, K0, K2 = self.form_reduced_canonical_matrices(
            **parameter_overrides)

        if len(M.shape) == 3:  # one of the parameters (not v or g) is an array
            A = np.zeros((M.shape[0], 4, 4))
            B = np.zeros((M.shape[0], 4, 2))
            for i, (Mi, C1i, K0i, K2i) in enumerate(zip(M, C1, K0, K2)):
                if 'g' in array_keys:
                    g = par['g'][i]
                else:
                    g = par['g']
                if 'v' in array_keys:
                    v = par['v'][i]
                else:
                    v = par['v']
                A[i], B[i] = ab_matrix(Mi, C1i, K0i, K2i, v, g)
        elif 'v' in array_keys or 'g' in array_keys:
            n = array_len
            A = np.zeros((n, 4, 4))
            B = np.zeros((n, 4, 2))
            for i in range(n):
                if 'g' in array_keys:
                    g = par['g'][i]
                else:
                    g = par['g']
                if 'v' in array_keys:
                    v = par['v'][i]
                else:
                    v = par['v']
                A[i], B[i] = ab_matrix(M, C1, K0, K2, v, g)
        else:  # scalar parameters
            A, B = ab_matrix(M, C1, K0, K2, par['v'], par['g'])

        return A, B

    def _calc_modal_controllability(self, acute=True, **parameter_overrides):
        """Returns the modal controllability [1]_ measures for each input and
        each eigenmode. The modal controllability is defined as the angle
        between each left eigenvector and each input column.

        Parameters
        ==========
        acute : boolean, optional
           If true only angles from 0 to pi/2 will be returned from the
           arrcos() computation. If false, angles from -pi/2 to pi/2 will be
           returned.
        **parameter_overrides : dictionary
            Parameter keys that map to floats or array_like of floats
            shape(n,). All keys that map to array_like must be of the same
            length.

        Returns
        =======
        beta : ndarray, shape(4,2) or shape(n, 4,2)
            Modal controllability angle for each eigenmode and each input.

        Notes
        =====

        ``x' = A*x + B*u``

        - A is n x n
        - B is n x m

        The columns of B are associated with the jth input:

        ``B = [b1, ..., bj, ..., bm]``

        The columns of Q are ith left eigenvectors of A, i.e. _, ``Q =
        eig(A.T)``:

        ``Q = [q1, ..., qi, ..., qn]``

        The modal controllability angle beta_ij is defined as::

           cos(beta_ij) = |qi.T @ bj|
                          -------------
                          ||qi||*||bj||

        - qi : ith left eigenvector
        - bj : jth column of B

        When qi is complex, the angle calculation is [2]_::

           cos(beta_ij) = Re(qi.H @ bj)
                          -------------
                          ||qi||*||bj||

        References
        ==========

        .. [1] A. M. A. Hamdan and A. H. Nayfeh, "Measures of modal
           controllability and observability for first- and second-order linear
           systems," Journal of Guidance, Control, and Dynamics, vol. 12, no.
           3, pp. 421–428, 1989, doi: 10.2514/3.20424.

        .. [2] https://en.wikipedia.org/wiki/Dot_product#Complex_vectors


        """

        _, B = self.form_state_space_matrices(**parameter_overrides)
        evals, evecs = self.calc_eigen(left=True, **parameter_overrides)
        evals, evecs = sort_eigenmodes(evals, evecs)

        def mod_cont(q, b, acute=True):
            """Returns the modal controllability value of the eigenvector q and
            input matrix column b.

            Parameters
            ==========
            q : array_like, shape(n,)
                A complex left eigenvector.
            b : array_like, shape(n,)
                A real column of the input matrix.

            Returns
            =======
            beta : float
                The (acute) angle in radians between q and b.

            """
            # vdot takes the complex conjugate of the first argument before
            # taking the dot product.
            num = np.real(np.vdot(q, b))  # Re(q.H @ b)
            # num = np.real(b @ np.conjugate(q).T)

            # norm_q = np.abs(np.sqrt(np.conjugate(q).T @ q))
            # norm_b = np.sqrt(b.T @ b)
            # den = norm_q*norm_b

            # norm() returns a real valued answer for the 2-norm
            den = np.linalg.norm(q)*np.linalg.norm(b)

            # NOTE : abs() forces 0 to 90 deg instead of 0 to 180 deg, i.e.
            # always the acute angle.
            if acute:
                cosbeta = np.abs(num/den)
            else:
                cosbeta = num/den
            return np.arccos(cosbeta)

        if len(B.shape) == 3:  # array version
            mod_ctrb = np.empty_like(B)
            for k, (Bk, vk) in enumerate(zip(B, evecs)):
                # columns of the evecs and columns of B
                for i, vi in enumerate(vk.T):
                    for j, bj in enumerate(Bk.T):
                        mod_ctrb[k, i, j] = mod_cont(vi, bj, acute=acute)
        else:
            mod_ctrb = np.empty((evecs.shape[1], B.shape[1]))
            for i, vi in enumerate(evecs.T):
                for j, bj in enumerate(B.T):
                    mod_ctrb[i, j] = mod_cont(vi, bj, acute=acute)

        return mod_ctrb

    def _plot_modal_controllability(self, axes=None, acute=True,
                                    **parameter_overrides):
        """Returns axes shape(4,2) with plots of the modal controllability for
        each input and each eigenmode."""

        par, array_keys, array_len = self._parse_parameter_overrides(
            **parameter_overrides)

        betas = self._calc_modal_controllability(acute=acute,
                                                 **parameter_overrides)
        betas = np.rad2deg(betas)

        if axes is None:
            fig, axes = plt.subplots(*betas[0].shape, sharex=True)

        for i, row in enumerate(axes):
            row[0].set_ylabel('Mode {}'.format(i + 1))
            for j, col in enumerate(row):
                col.plot(par[array_keys[0]], betas[:, i, j])

        axes[3, 0].set_xlabel(array_keys[0])
        axes[3, 1].set_xlabel(array_keys[0])

        axes[0, 0].set_title(r'Input: $T_\phi$')
        axes[0, 1].set_title(r'Input: $T_\delta$')

        return axes

    def plot_simulation(self, times, initial_conditions, input_func=None,
                        **parameter_overrides):
        """Returns the state and input trajectories at each time value.

        Parameters
        ==========
        times : array_like, shape(n,)
            Monotonic increasing time values to simulate over.
        initial_conditions : array_like, shape(4,)
            Initial values of the states.
        input_func : function
            Takes form u = f(t, x) where u is array_like, shape(2,).
        **parameter_overrides : dictionary
            Parameter keys that map to floats or array_like of floats
            shape(n,). All keys that map to array_like must be of the same
            length.

        Returns
        =======
        axes : ndarray, shape(3,)
            Three subplots that plot the input trajectories, state angle
            trajectories, and state angular rates.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           import numpy as np
           from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
           from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
           from bicycleparameters.models import Meijaard2007Model
           p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
           m = Meijaard2007Model(p)
           times = np.linspace(0.0, 5.0, num=51)
           x0 = np.deg2rad([10.0, 5.0, 0.0, 0.0])
           m.plot_simulation(times, x0, v=6.0)

        """
        res, inputs = self.simulate(times, initial_conditions,
                                    input_func=input_func,
                                    **parameter_overrides)

        fig, axes = plt.subplots(3, sharex=True, layout='constrained')

        axes[0].plot(times, inputs)
        axes[0].legend([r'$T_\phi$', r'$T_\delta$'])
        axes[0].set_ylabel('Torque\n[Nm]')
        axes[1].plot(times, np.rad2deg(res[:, :2]))
        axes[1].legend(['$' + lab + '$' for lab in self.state_vars_latex[:2]])
        axes[1].set_ylabel('Angle\n[deg]')
        axes[2].plot(times, np.rad2deg(res[:, 2:]))
        axes[2].legend(['$' + lab + '$' for lab in self.state_vars_latex[2:]])
        axes[2].set_ylabel('Angluar Rate\n[deg/s]')

        axes[2].set_xlabel('Time [s]')

        return axes


class Meijaard2007WithFeedbackModel(Meijaard2007Model):
    """Linear Carvallo-Whipple bicycle model that includes full state feedback
    to drive all states to zero. With two inputs (roll torque and steer torque)
    and four states (roll angle, steer angle, roll rate, steer rate) there are
    eight control gain parameters in addition to the parameters defined in
    [Meijaard2007]_.

    The states are::

       x = |roll angle         | = |phi     |
           |steer angle        |   |delta   |
           |roll angular rate  |   |phidot  |
           |steer angular rate |   |deltadot|

    The inputs are::

       u = |roll torque | = |Tphi  |
           |steer torque|   |Tdelta|

    Applying full state feedback gives this controller::

       u = -K*x = -|kTphi_phi, kTphi_del, kTphi_phid, kTphi_deld|*|phi     |
                   |kTdel_phi, kTdel_del, kTdel_phid, kTdel_deld| |delta   |
                                                                  |phidot  |
                                                                  |deltadot|

    This represents the new model::

       x' = (A - B*K)*x + B*u

    so steer and roll torque can be applied in parallel to the feedback
    control.

    """
    def __init__(self, parameter_set):
        self.parameter_set = parameter_set.to_parameterization(
            'Meijaard2007WithFeedback')

    def form_state_space_matrices(self, **parameter_overrides):
        """Returns the A and B matrices for the Carvallo-Whipple model
        linearized about the upright constant velocity configuration with a
        full state feedback steer controller to drive the states to zero.

        Returns
        =======
        A : ndarray, shape(4,4) or shape(n,4,4)
            The state matrix.
        B : ndarray, shape(4,2) or shape(n,4,2)
            The input matrix.

        Notes
        =====

        A, B, and K describe the model in state space form::

           x' = (A - B*K)*x + B*u

        where::

           x = |phi     | = |roll angle         |
               |delta   |   |steer angle        |
               |phidot  |   |roll angular rate  |
               |deldot  |   |steer angular rate |

           K = | kTphi_phi kTphi_del kTphi_phid kTphi_deld |
               | kTdel_phi kTdel_del kTdel_phid kTdel_deld |

           u = |Tphi  | = |roll torque |
               |Tdelta|   |steer torque|

        """
        gain_names = ['kTphi_phi', 'kTphi_del', 'kTphi_phid', 'kTphi_deld',
                      'kTdel_phi', 'kTdel_del', 'kTdel_phid', 'kTdel_deld']

        par, arr_keys, arr_len = self._parse_parameter_overrides(
            **parameter_overrides)

        # g, v, and the contoller gains are not used in the computation of M,
        # C1, K0, K2.

        M, C1, K0, K2 = self.form_reduced_canonical_matrices(
            **parameter_overrides)

        # steer controller gains, 2x4, no roll control
        if any(k in gain_names for k in arr_keys):
            # if one of the gains is an array, create a set of gain matrices
            # where that single gain varies across the set
            K = np.array([
                [par[p][0] if p in arr_keys else par[p] for p in gain_names[:4]],
                [par[p][0] if p in arr_keys else par[p] for p in gain_names[4:]]
            ])
            # K is now shape(n, 2, 4)
            K = np.tile(K, (arr_len, 1, 1))
            for k in arr_keys:
                if k in gain_names[:4]:
                    K[:, 0, gain_names[:4].index(k)] = par[k]
                if k in gain_names[4:]:
                    K[:, 1, gain_names[4:].index(k)] = par[k]
        else:  # gains are not an array
            K = np.array([[par[p] for p in gain_names[:4]],
                          [par[p] for p in gain_names[4:]]])

        if arr_keys:
            A = np.zeros((arr_len, 4, 4))
            B = np.zeros((arr_len, 4, 2))
            for i in range(arr_len):
                Mi = M[i] if M.ndim == 3 else M
                C1i = C1[i] if C1.ndim == 3 else C1
                K0i = K0[i] if K0.ndim == 3 else K0
                K2i = K2[i] if K2.ndim == 3 else K2
                vi = par['v'] if np.isscalar(par['v']) else par['v'][i]
                gi = par['g'] if np.isscalar(par['g']) else par['g'][i]
                Ki = K[i] if K.ndim == 3 else K
                Ai, Bi = ab_matrix(Mi, C1i, K0i, K2i, vi, gi)
                A[i] = Ai - Bi@Ki
                B[i] = Bi
        else:  # scalar parameters
            A, B = ab_matrix(M, C1, K0, K2, par['v'], par['g'])
            A = A - B@K

        return A, B

    def plot_gains(self, axes=None, **parameter_overrides):
        """Plots the gains versus a single varying parameter. The
        ``parameter_overrides`` should contain one parameter that is an array,
        other than the eight gains. That parameter will be used for the x axis.
        The gains can be either arrays of the same length or scalars.

        Parameters
        ==========
        axes : array_like, shape(2, 4)
            Matplotlib axes set to plot to.
        parameter_overrides : dictionary
            Parameter keys that map to floats or array_like of floats
            shape(n,). All keys that map to array_like must be of the same
            length.

        Returns
        =======
        axes : ndarray, shape(2, 4)
            Array of matplotlib axes.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           import numpy as np
           from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
           from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
           from bicycleparameters.models import Meijaard2007WithFeedbackModel
           p = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
           m = Meijaard2007WithFeedbackModel(p)
           m.plot_gains(v=np.linspace(0.0, 10.0, num=101),
                        kTdel_phid=-10.0*np.linspace(0.0, 5.0, num=101))

        """
        gain_names = ['kTphi_phi', 'kTphi_del', 'kTphi_phid', 'kTphi_deld',
                      'kTdel_phi', 'kTdel_del', 'kTdel_phid', 'kTdel_deld']

        if axes is None:
            fig, axes = plt.subplots(2, 4, sharex=True, layout='constrained')

        par, array_keys, array_len = self._parse_parameter_overrides(
            **parameter_overrides)

        non_gain_array_keys = [k for k in array_keys if k not in gain_names]
        if len(non_gain_array_keys) == 0:
            raise ValueError('No x axis key. Set one parameter other than the '
                             'gains as an array.')
        else:
            x_axis_key = non_gain_array_keys[0]

        if len(non_gain_array_keys) > 1:
            msg = 'More than one array for x axis, choosing {}.'
            print(msg.format(x_axis_key))

        for ax, name in zip(axes.flatten(), gain_names):
            if name in array_keys:
                ax.plot(par[x_axis_key], par[name])
            else:
                ax.plot(par[x_axis_key],
                        par[name]*np.ones_like(par[x_axis_key]))
            msg = r'${}$'.format(self.parameter_set.par_strings[name])
            ax.set_title(msg)

        for ax in axes[1, :]:
            ax.set_xlabel(x_axis_key)

        return axes


class MooreRiderLean2012Model(_Model):
    """Carvallo-Whipple model with an extra rigid body that represents the
    rider's upper body which can lean relative to the bicycle's rear frame.

    This specific model is described in [Moore2012]_ and the equations of
    motion linearized about the nominal upright configuration at a constant
    longitudinal speed were extracted from the source code from the
    dissertation and integrated here.

    **12 States**:

    - :math:`q_1`, 0: First Cartesian coordinate in the ground plane to rear
      wheel contact [m]
    - :math:`q_2`, 1: Second Cartesian coordinate in the ground plane to rear
      wheel contact [m]
    - :math:`q_3`, 2: rear frame yaw angle [rad]
    - :math:`q_4`, 3: rear frame roll angle [rad]
    - :math:`q_6`, 4: rear wheel angle relative to rear frame [rad]
    - :math:`q_7`, 5: steer angle (front frame relative to rear frame) [rad]
    - :math:`q_8`, 6: front wheel angle relative to front frame [rad]
    - :math:`q_9`, 7: rider lean angle relative to rear frame [rad]
    - :math:`u_4`, 8: rear frame roll rate [rad/s]
    - :math:`u_6`: rear wheel angular rate relative to rear frame [rad/s]
    - :math:`u_7`, 9: steer angular rate relative to the rear frame [rad/s]
    - :math:`u_9`, 10: rider lean angular rate relative to the rear frame
      [rad/s]

    **3 Inputs**:

    - :math:`T_4`, 0: roll torque (between rear frame and ground) [Nm]
    - :math:`T_7`, 1: steer torque (between rear and front frames [Nm]
    - :math:`T_9`, 2: rider lean torque (between rider upper body and rear
      frame) [Nm]

    """
    input_vars = ['T4', 'T7', 'T9']
    state_vars = ['q1', 'q2', 'q3', 'q4', 'q6', 'q7', 'q8', 'q9',
                  'u4', 'u6', 'u7', 'u9']
    input_units = ['N-m', 'N-m', 'N-m']
    state_units = ['m', 'm', 'rad', 'rad', 'rad', 'rad', 'rad', 'rad',
                   'rad/s', 'rad/s', 'rad/s', ]
    input_vars_latex = ['T_4', 'T_7', 'T_9']
    state_vars_latex = ['q_1', 'q_2', 'q_3', 'q_4', 'q_6', 'q_7', 'q_8', 'q_9',
                        'u_4', 'u_6', 'u_7', 'u_9']

    def __init__(self, parameter_set):

        # NOTE : this import is here because it is a bit slow loading into
        # memory, so it only loads if someone uses this class.
        from bicycleparameters._mooreriderlean2012 import eval_linear

        self._eval_linear = eval_linear

        self.parameter_set = parameter_set.to_parameterization(
            'MooreRiderLean2012')

    def form_state_space_matrices(self, **parameter_overrides):
        """Returns the A and B matrices for the model linearized about the
        upright constant longitudinal speed configuration.

        Parameters
        ==========
        **parameter_overrides : dictionary
            Parameter keys that map to floats or array_like of floats
            shape(n,). All keys that map to array_like must be of the same
            length.

        Returns
        =======
        A : ndarray, shape(12,12) or shape(n,12,12)
            The state matrix.
        B : ndarray, shape(12,3) or shape(n,12,3)
            The input matrix.

        Notes
        =====

        ``A`` and ``B`` describe the model in state space form:

        ``x' = A * x + B * u``

        Examples
        ========

        >>> from bicycleparameters.parameter_dicts import mooreriderlean2012_browser_jason
        >>> from bicycleparameters.parameter_sets import MooreRiderLean2012ParameterSet
        >>> from bicycleparameters.models import MooreRiderLean2012Model
        >>> p = MooreRiderLean2012ParameterSet(mooreriderlean2012_browser_jason)
        >>> m = MooreRiderLean2012Model(p)
        >>> A, B = m.form_state_space_matrices(v=4.0)
        >>> A
        array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00, -8.26759379e-33,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00, -0.00000000e+00,
                -3.40958859e-01, -0.00000000e+00,  0.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  4.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                -0.00000000e+00, -0.00000000e+00,  0.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  3.28701304e+00,
                 0.00000000e+00,  0.00000000e+00, -0.00000000e+00,
                 0.00000000e+00,  5.63565404e-02,  0.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                -0.00000000e+00,  3.30703751e-32,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00, -0.00000000e+00,
                 9.92516037e-01, -0.00000000e+00,  0.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00, -0.00000000e+00,
                 1.11095692e+01,  0.00000000e+00, -1.52341011e+01,
                -0.00000000e+00, -8.46683843e+00, -2.04638864e-01,
                 0.00000000e+00, -1.34590803e+00,  0.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00, -0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00, -0.00000000e+00,
                -0.00000000e+00, -0.00000000e+00, -0.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00, -0.00000000e+00,
                 1.04939762e+01,  0.00000000e+00,  5.61738098e+00,
                -0.00000000e+00,  1.68341653e+01,  8.26892076e+00,
                 0.00000000e+00, -1.01980345e+01,  0.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                -1.47851073e+01, -0.00000000e+00,  2.03755703e+01,
                 0.00000000e+00,  5.11495540e+01,  4.37754540e-01,
                -0.00000000e+00,  2.54094086e+00, -0.00000000e+00]])
        >>> B
        array([[ 0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ],
               [ 0.02883808, -0.11361237, -0.09393174],
               [ 0.        , -0.        , -0.        ],
               [-0.11361237,  4.59077823,  0.24303462],
               [-0.09393174,  0.24303462,  0.48717313]])

        """
        par, array_keys, array_len = self._parse_parameter_overrides(
            **parameter_overrides)

        mutable_par = par.copy()

        q, u, r = np.zeros(9), np.zeros(4), np.zeros(4)

        v = mutable_par.pop('v')

        # NOTE : eval_linear returns the A (13, 13) and B (13, 4) but we can
        # exclude q5, T6. These are the indices to keep.
        a_idxs = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]
        b_idxs = [0, 2, 3]

        if array_keys:

            A = np.zeros((array_len, 12, 12))
            B = np.zeros((array_len, 12, 3))

            vi = v

            for i in range(array_len):
                for k in array_keys:
                    if k == 'v':
                        vi = v[i]
                    else:
                        try:
                            mutable_par[k] = par[k][i]
                        except TypeError:
                            mutable_par[k] = par[k]
                q[4] = pitch_from_roll_and_steer(q[3], q[6],
                                                 mutable_par['rf'],
                                                 mutable_par['rr'],
                                                 mutable_par['d1'],
                                                 mutable_par['d2'],
                                                 mutable_par['d3'])
                u[1] = -vi/mutable_par['rr']  # u6
                par_arr = np.array([
                    mutable_par[k]
                    for k in self.parameter_set.par_strings.keys()
                    if k in mutable_par.keys()
                ])
                A_, B_, _, _ = self._eval_linear(q, u, r, par_arr)
                A[i] = A_[a_idxs][:, a_idxs]
                B[i] = B_[a_idxs][:, b_idxs]
        else:  # scalar parameters
            q[4] = pitch_from_roll_and_steer(q[3], q[6],
                                             mutable_par['rf'],
                                             mutable_par['rr'],
                                             mutable_par['d1'],
                                             mutable_par['d2'],
                                             mutable_par['d3'])
            u[1] = -v/mutable_par['rr']  # u6
            par_arr = np.array([
                mutable_par[k]
                for k in self.parameter_set.par_strings.keys()
                if k in mutable_par.keys()
            ])
            A_, B_, _, _ = self._eval_linear(q, u, r, par_arr)
            A = np.zeros((12, 12))
            B = np.zeros((12, 3))
            A, B = A_[a_idxs][:, a_idxs], B_[a_idxs][:, b_idxs]

        return A, B

    def simulate(self, times, initial_conditions, input_func=None,
                 **parameter_overrides):
        par, array_keys, array_len = self._parse_parameter_overrides(
            **parameter_overrides)

        # u6 has to be consistent with v
        initial_conditions[9] = -par['v']/par['rr']

        return super().simulate(times, initial_conditions,
                                input_func=input_func, **parameter_overrides)

    def plot_simulation(self, times, initial_conditions, input_func=None,
                        **parameter_overrides):
        """Returns the state and input trajectories at each provided time
        value.

        Parameters
        ==========
        times : array_like, shape(m,)
            Monotonic increasing time values to simulate over.
        initial_conditions : array_like, shape(12,)
            Initial values of the states: ``[q1, q2, q3, q4, q6, q7, q8, q9,
            u4, u6, u7, u9]``.
        input_func : function
            Takes form ``r = f(t, x)`` where ``r`` is array_like, shape(4,).
            with ``r = [T4, T7, T9]``.
        **parameter_overrides : dictionary
            Parameter keys that map to floats or array_like of floats
            shape(n,). All keys that map to array_like must be of the same
            length.

        Returns
        =======
        axes : ndarray, shape(4,)
            Four subplots that plot the input trajectories, distance
            trajectories, state angle trajectories, and state angular rates.

        Examples
        ========

        .. plot::
           :include-source: True
           :context: reset

           import numpy as np

           from bicycleparameters.parameter_dicts import mooreriderlean2012_browser_jason
           from bicycleparameters.parameter_sets import MooreRiderLean2012ParameterSet
           from bicycleparameters.models import MooreRiderLean2012Model
           p = MooreRiderLean2012ParameterSet(mooreriderlean2012_browser_jason)
           m = MooreRiderLean2012Model(p)
           times = np.linspace(0.0, 5.0, num=51)
           x0 = np.zeros(12)
           x0[5] = np.deg2rad(5.0)  # steer angle
           m.plot_simulation(times, x0, v=20.0)

        """
        par, array_keys, array_len = self._parse_parameter_overrides(
            **parameter_overrides)

        # u6 has to be consistent with v
        initial_conditions[9] = -par['v']/par['rr']

        res, inputs = self.simulate(times, initial_conditions,
                                    input_func=input_func,
                                    **parameter_overrides)

        fig, axes = plt.subplots(5, sharex=True, layout='constrained')

        # roll, steer, lean torques
        axes[0].plot(times, inputs)
        labs = ['$' + lab + '$' for lab in self.input_vars_latex]
        axes[0].legend(labs, ncol=5)
        axes[0].set_ylabel('Torque\n[Nm]')

        # x, y of rear wheel contact
        axes[1].plot(times, res[:, :2])
        labs = ['$' + lab + '$' for lab in self.state_vars_latex[:2]]
        axes[1].legend(labs, ncol=5)
        axes[1].set_ylabel('Distance\n[m]')

        # wheel angles
        idxs = [4, 6]
        axes[2].plot(times, np.rad2deg(res[:, idxs]))
        labs = ['$' + self.state_vars_latex[i] + '$' for i in idxs]
        axes[2].legend(labs, ncol=5)
        axes[2].set_ylabel('Angle\n[deg]')

        # roll, steer, lean angles
        idxs = [3, 5, 7]
        axes[3].plot(times, np.rad2deg(res[:, idxs]))
        labs = ['$' + self.state_vars_latex[i] + '$' for i in idxs]
        axes[3].set_ylabel('Angle\n[deg]')

        # yaw
        ax = axes[3].twinx()
        ax.plot(times, np.rad2deg(res[:, 2]), color='C4')
        ax.set_ylabel('$' + self.state_vars_latex[2] + '$ [deg]')

        axes[3].legend(labs, ncol=5)

        # roll, steer, lean rates
        idxs = [8, 10, 11]
        axes[4].plot(times, np.rad2deg(res[:, idxs]))
        labs = ['$' + self.state_vars_latex[i] + '$' for i in idxs]
        axes[4].set_ylabel('Angluar Rate\n[deg/s]')
        axes[4].set_xlabel('Time [s]')

        # wheel speed
        ax = axes[4].twinx()
        ax.plot(times, np.rad2deg(res[:, 9]), color='C3')
        ax.set_ylabel('$' + self.state_vars_latex[9] + '$ [deg/s]')

        axes[4].legend(labs, ncol=5)

        return axes
