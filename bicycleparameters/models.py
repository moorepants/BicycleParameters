import warnings

import numpy as np
import matplotlib.pyplot as plt

from .bicycle import benchmark_par_to_canonical, ab_matrix, sort_eigenmodes


class Meijaard2007Model(object):
    """Whipple-Carvallo model presented in [Meijaard2007]_. It is both linear
    and the minimal model in terms of states and coordinates that fully
    describe the vehicles dynamics.

    References
    ==========

    .. [Meijaard2007] Meijaard J.P, Papadopoulos Jim M, Ruina Andy and Schwab
       A.L, 2007, Linearized dynamics equations for the balance and steer of a
       bicycle: a benchmark and review, Proc. R. Soc. A., 463:1955–1982
       http://doi.org/10.1098/rspa.2007.1857

    """
    state_vars_latex = [r'\phi', r'\delta', r'\dot{\phi}', r'\dot{\delta}']

    def __init__(self, benchmark_parameter_set):
        """Initializes the model with the provided parameters.

        Parameters
        ==========
        benchmark_parameters : dictionary
            Dictionary that maps floats to the parameter keys containing:

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

        """
        self.parameter_set = benchmark_parameter_set

    def _parse_parameter_overrides(self, **parameter_overrides):

        par = self.parameter_set.parameters.copy()

        found_one = False
        array_key = None
        array_val = None
        for key, val in parameter_overrides.items():
            if key not in par.keys():  # don't add invalid keys
                msg = '{} is not a valid parameter, ignoring'
                warnings.warn(msg.format(key))
            else:
                # TODO : if the array is a list this may not work
                if np.isscalar(val):
                    par[key] = float(val)
                else:  # is an array
                    # TODO : It would be useful if more than one can be an
                    # array. As long as the arrays are the same length you
                    # could doe this. For example this is helpful for gain
                    # scheduling multiple gains across speeds.
                    if found_one:
                        msg = 'Only 1 parameter can be an array of values.'
                        raise ValueError(msg)
                    # pass v and g through if arrays
                    elif key == 'v' or key == 'g':
                        par[key] = val
                        array_key = key
                        found_one = True
                    else:
                        array_key = key
                        array_val = val
                        par[key] = val
                        found_one = True

        return par, array_key, array_val

    def form_reduced_canonical_matrices(self, **parameter_overrides):
        """Returns the canonical speed and gravity independent matrices for the
        Whipple-Carvallo bicycle model linearized about the nominal
        configuration.

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

            M*q'' + v*C1*q' + [g*K0 + v**2*K2]*q = f

        where:

            q = [phi, delta]
            f = [Tphi, Tdelta]

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

        """
        par, array_key, array_val = self._parse_parameter_overrides(
            **parameter_overrides)
        canon_deps = ['IBxx', 'IBxz', 'IByy', 'IBzz', 'IFxx', 'IFyy', 'IHxx',
                      'IHxz', 'IHyy', 'IHzz', 'IRxx', 'IRyy', 'c', 'lam', 'mB',
                      'mF', 'mH', 'mR', 'rF', 'rR', 'w', 'xB', 'xH', 'zB',
                      'zH']

        if array_val is not None and array_key in canon_deps:
            M = np.zeros((len(array_val), 2, 2))
            C1 = np.zeros((len(array_val), 2, 2))
            K0 = np.zeros((len(array_val), 2, 2))
            K2 = np.zeros((len(array_val), 2, 2))
            for i, val in enumerate(array_val):
                par[array_key] = val
                M[i], C1[i], K0[i], K2[i] = benchmark_par_to_canonical(par)
            return M, C1, K0, K2
        else:
            return benchmark_par_to_canonical(par)

    def form_state_space_matrices(self, **parameter_overrides):
        """Returns the A and B matrices for the Whipple model linearized about
        the upright constant velocity configuration.

        Parameters
        ==========
        speed : float
            The speed of the bicycle.

        Returns
        =======
        A : ndarray, shape(4,4)
            The state matrix.
        B : ndarray, shape(4,2)
            The input matrix.

        Notes
        =====
        ``A`` and ``B`` describe the Whipple model in state space form:

            x' = A * x + B * u

        where

        The states are [roll angle,
                        steer angle,
                        roll rate,
                        steer rate]

        The inputs are [roll torque,
                        steer torque]

        """
        if 'g' in parameter_overrides.keys():
            g = parameter_overrides['g']
        else:
            g = self.parameter_set.parameters['g']

        if 'v' in parameter_overrides.keys():
            v = parameter_overrides['v']
        else:
            v = self.parameter_set.parameters['v']

        M, C1, K0, K2 = self.form_reduced_canonical_matrices(
            **parameter_overrides)

        if len(M.shape) == 3:  # one of the parameters (not v or g) is an array
            A = np.zeros((M.shape[0], 4, 4))
            B = np.zeros((M.shape[0], 4, 2))
            for i, (Mi, C1i, K0i, K2i) in enumerate(zip(M, C1, K0, K2)):
                A[i], B[i] = ab_matrix(Mi, C1i, K0i, K2i, v, g)
        elif not isinstance(v, float):
            A = np.zeros((len(v), 4, 4))
            B = np.zeros((len(v), 4, 2))
            for i, vi in enumerate(v):
                A[i], B[i] = ab_matrix(M, C1, K0, K2, vi, g)
        elif not isinstance(g, float):
            A = np.zeros((len(g), 4, 4))
            B = np.zeros((len(g), 4, 2))
            for i, gi in enumerate(g):
                A[i], B[i] = ab_matrix(M, C1, K0, K2, v, gi)
        else:  # scalar parameters
            A, B = ab_matrix(M, C1, K0, K2, v, g)

        return A, B

    def calc_eigen(self, left=False, **parameter_overrides):
        """Returns the eigenvalues and eigenvectors of the model.

        Parameters
        ==========
        left : boolean, optional
            If true, the left eigenvectors will be returned, i.e. A.T*v=lam*v.

        Returns
        =======
        evals : ndarray, shape(4,) or shape (n, 4)
            eigenvalues
        evecs : ndarray, shape(4,4) or shape (n, 4, 4)
            eigenvectors

        """
        A, B = self.form_state_space_matrices(**parameter_overrides)

        if len(A.shape) == 3:  # array version
            evals = np.zeros(A.shape[:2], dtype='complex128')
            evecs = np.zeros(A.shape, dtype='complex128')
            for i, Ai in enumerate(A):
                if left:
                    Ai = Ai.T
                evals[i], evecs[i] = np.linalg.eig(Ai)
            return evals, evecs
        else:
            if left:
                A = A.T
            return np.linalg.eig(A)

    def calc_modal_controllability(self, **parameter_overrides):
        """Returns the modal controllability measures.

        cos(beta_ij) = |qi.T @ bj|
                       -------------
                       ||qi|| ||bj||

        qi : ith left eigenvector
        bj : jth column of B

        A. M. A. Hamdan and A. H. Nayfeh, "Measures of modal controllability
        and observability for first- and second-order linear systems," Journal
        of Guidance, Control, and Dynamics, vol. 12, no. 3, pp. 421–428, 1989,
        doi: 10.2514/3.20424.


        When q is complex, the angle calculation is
        https://en.wikipedia.org/wiki/Dot_product#Complex_vectors

        cos(beta_ij) = Re(qi.H @ bj)
                       -------------
                       ||qi|| ||bj||
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
            #num = np.real(b @ np.conjugate(q).T)

            #norm_q = np.abs(np.sqrt(np.conjugate(q).T @ q))
            #norm_b = np.sqrt(b.T @ b)
            #den = norm_q*norm_b

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
                        mod_ctrb[k, i, j] = mod_cont(vi, bj)
        else:
            mod_ctrb = np.empty((evecs.shape[1], B.shape[1]))
            for i, vi in enumerate(evecs.T):
                for j, bj in enumerate(B.T):
                    mod_ctrb[i, j] = mod_cont(vi, bj)

        return mod_ctrb

    def plot_eigenvalue_parts(self, ax=None, colors=None,
                              **parameter_overrides):
        """Returns a Matplotlib axis of the real and imaginary parts of the
        eigenvalues plotted against the provided parameter.

        Parameters
        ==========
        ax : Axes
            Matplotlib axes.
        colors : sequence, len(4)
            Matplotlib colors for the 4 modes.

        """

        if ax is None:
            fig, ax = plt.subplots()

        evals, evecs = self.calc_eigen(**parameter_overrides)
        if len(evals.shape) > 1:
            evals, evecs = sort_eigenmodes(evals, evecs)
        else:
            evals, evecs = [evals], [evecs]

        par, array_key, array_val = self._parse_parameter_overrides(
            **parameter_overrides)

        if colors is None:
            colors = ['C0', 'C1', 'C2', 'C3']
        legend = ['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4',
                  'Mode 1', 'Mode 2', 'Mode 3', 'Mode 4']
        # imaginary components
        for eval_sequence, color, label in zip(evals.T, colors, legend):
            ax.plot(par[array_key], np.imag(eval_sequence),
                    color=color, label=label, linestyle='--')

        # x axis line
        #ax.plot(speeds, np.zeros_like(speeds), 'k-', label='_nolegend_',
                #linewidth=1.5)

        # plot the real parts of the eigenvalues
        for eval_sequence, color, label in zip(evals.T, colors, legend):
            ax.plot(par[array_key], np.real(eval_sequence), color=color,
                    label=label)

        # set labels and limits
        ax.set_ylabel('Real and Imaginary Parts of the Eigenvalue [1/s]')
        ax.set_xlim((par[array_key][0], par[array_key][-1]))

        ax.grid()

        ax.set_xlabel(array_key)

        return ax

    def plot_eigenvectors(self, **parameter_overrides):
        """Plots the components of the eigenvectors in the real and imaginary
        plane.

        Parameters
        ----------

        Returns
        -------
        figs : list
            A list of matplotlib figures.
        Notes
        -----
        Plots are not produced for zero eigenvalues.
        """
        par, array_key, array_val = self._parse_parameter_overrides(
            **parameter_overrides)
        if array_key in ['v', 'g']:
            array_val = par[array_key]
        states = [r'\phi', r'\delta', r'\dot{\phi}', r'\dot{\delta}']
        eval_seq, evec_seq = self.calc_eigen(**parameter_overrides)
        eval_seq = np.atleast_2d(eval_seq)
        evec_seq = np.atleast_2d(evec_seq)
        # TODO : not needed if no varyied params
        eval_seq, evec_seq = sort_eigenmodes(eval_seq, evec_seq)
        fig, axes = plt.subplots(*eval_seq.shape,
                                 subplot_kw={'projection': 'polar'})
        axes = np.atleast_2d(axes)
        lw = list(range(1, len(states) + 1))
        lw.reverse()
        for k, (evals, par_val) in enumerate(zip(eval_seq, array_val)):
            axes[k, 0].set_ylabel('{} = {}'.format(array_key, par_val))
            for i, eVal in enumerate(evals):
                eVec = evec_seq[k, :, i]
                maxCom = abs(eVec[:2]).max()
                for j, component in enumerate(eVec[:2]):
                    radius = abs(component) / maxCom
                    theta = np.angle(component)
                    axes[k, i].plot([0, theta], [0, radius], lw=lw[j])
                axes[k, i].set_rmax(1.0)
                axes[k, i].legend(['$' + s + '$' for s in states])
                axes[k, i].set_title('Eigenvalue: %1.3f$\pm$%1.3fj' % (eVal.real, eVal.imag))

        #fig.tight_layout()

        return axes
