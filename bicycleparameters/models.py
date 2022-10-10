import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

from .bicycle import benchmark_par_to_canonical, ab_matrix, sort_eigenmodes


class Meijaard2007Model(object):
    """Whipple-Carvallo model presented in [Meijaard2007]_. It is both linear
    and the minimal model in terms of states and coordinates that fully
    describe the vehicles dynamics: self-stability and non-minimum phase
    behavior.

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

    def plot_modal_controllability(self, **parameter_overrides):

        par, array_keys, array_len = self._parse_parameter_overrides(
            **parameter_overrides)

        betas = self.calc_modal_controllability(**parameter_overrides)
        betas = np.rad2deg(betas)
        fig, axes = plt.subplots(*betas[0].shape, sharex=True, sharey=True)
        axes[0, 0].plot(par[array_keys[0]], betas[:, 0, 0])
        axes[0, 1].plot(par[array_keys[0]], betas[:, 0, 1])
        axes[1, 0].plot(par[array_keys[0]], betas[:, 1, 0])
        axes[1, 1].plot(par[array_keys[0]], betas[:, 1, 1])
        axes[2, 0].plot(par[array_keys[0]], betas[:, 2, 0])
        axes[2, 1].plot(par[array_keys[0]], betas[:, 2, 1])
        axes[3, 0].plot(par[array_keys[0]], betas[:, 3, 0])
        axes[3, 1].plot(par[array_keys[0]], betas[:, 3, 1])

        return axes

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

        par, array_keys, _ = self._parse_parameter_overrides(
            **parameter_overrides)

        if colors is None:
            colors = ['C0', 'C1', 'C2', 'C3']
        legend = ['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4',
                  'Mode 1', 'Mode 2', 'Mode 3', 'Mode 4']
        # imaginary components
        for eval_sequence, color, label in zip(evals.T, colors, legend):
            ax.plot(par[array_keys[0]], np.imag(eval_sequence),
                    color=color, label=label, linestyle='--')

        # plot the real parts of the eigenvalues
        for eval_sequence, color, label in zip(evals.T, colors, legend):
            ax.plot(par[array_keys[0]], np.real(eval_sequence), color=color,
                    label=label)

        # set labels and limits
        ax.set_ylabel('Real and Imaginary Parts of the Eigenvalue [1/s]')
        ax.set_xlim((par[array_keys[0]][0], par[array_keys[0]][-1]))

        ax.grid()

        ax.set_xlabel(array_keys[0])

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
        par, arr_keys, _ = self._parse_parameter_overrides(
            **parameter_overrides)
        states = [r'\phi', r'\delta', r'\dot{\phi}', r'\dot{\delta}']

        eval_seq, evec_seq = self.calc_eigen(**parameter_overrides)
        eval_seq, evec_seq = np.atleast_2d(eval_seq), np.atleast_3d(evec_seq)
        # TODO : sort_eigenmodes() is doing something funny and not adding the
        # 4th eigenval, so you often end up with duplicates eigenvalues for one
        # eigenval and one missing. Also the algorithm may not work well with
        # spaced out eigenvalues, which is what I've been trying here. You may
        # have to calculate eigenvals/vec across closer spacing, then sample
        # out the ones you want. For now, we don't sort coarse spaced
        # eigenvalues.
        #if arr_keys:
            #eval_seq, evec_seq = sort_eigenmodes(eval_seq, evec_seq)

        fig, axes = plt.subplots(*eval_seq.shape,
                                 subplot_kw={'projection': 'polar'})
        axes = np.atleast_2d(axes)
        fig.set_size_inches(axes.shape[1]*3, axes.shape[0]*3)
        lw = list(range(1, len(states) + 1))
        lw.reverse()

        for k, (evals, evecs, par_val) in enumerate(zip(eval_seq, evec_seq,
                                                        par[arr_keys[0]])):

            axes[k, 0].set_ylabel('{} = {:1.2f}'.format(arr_keys[0], par_val),
                                  labelpad=30)

            for i, (eigenval, eigenvec) in enumerate(zip(evals, evecs.T)):

                max_com = np.abs(eigenvec[:2]).max()

                for j, component in enumerate(eigenvec[:2]):

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

        axes[0, 0].legend(['$' + s + '$' for s in states],
                          loc='upper center', bbox_to_anchor=(0.5, 1.05),
                          fancybox=True, shadow=True, ncol=4)

        fig.tight_layout()

        return axes

    def simulate(self, times, initial_conditions, input_func=None,
                 **parameter_overrides):
        """Returns the

        input_func : function
            Takes form f(t, x, par).
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
                return A@x + B@input_func(t, x, par)

        res = spi.solve_ivp(eval_rhs,
                            (times[0], times[-1]),
                            initial_conditions,
                            t_eval=times)

        if input_func is None:
            inputs = np.zeros((len(times), 2))
        else:
            inputs = np.empty((len(times), 2))
            for i, ti in enumerate(times):
                ui = input_func(ti, res.y[:, i], par)
                inputs[i, :] = ui[:]

        return res.y.T, inputs

    def plot_simulation(self, times, initial_conditions, input_func=None,
                        **parameter_overrides):

        res, inputs = self.simulate(times, initial_conditions,
                                    input_func=input_func,
                                    **parameter_overrides)

        fig, axes = plt.subplots(3, sharex=True)

        axes[0].plot(times, inputs)
        axes[0].legend([r'$T_\phi$', r'$T_\delta$'])
        axes[1].plot(times, np.rad2deg(res[:, :2]))
        axes[1].legend(['$' + lab + '$' for lab in self.state_vars_latex[:2]])
        axes[2].plot(times, np.rad2deg(res[:, 2:]))
        axes[2].legend(['$' + lab + '$' for lab in self.state_vars_latex[2:]])

        axes[2].set_xlabel('Time [s]')

        return axes

    def simulate_modes(self, **parameter_overrides):

        par, arr_keys, _ = self._parse_parameter_overrides(
            **parameter_overrides)

        if arr_keys:
            raise ValueError('Can only simulate with fixed parameters.')

        A, B = self.form_state_space_matrices(**parameter_overrides)
        evals, evecs = self.calc_eigen(**parameter_overrides)

        def eval_rhs(t, x):
            return A@x

        times = np.linspace(0.0, 10.0, num=1000)

        results = np.empty((4, len(times), 4))

        for i, evec in enumerate(evecs.T):
            initial_condition = evec.real

            sim_res = spi.solve_ivp(eval_rhs, (times[0], times[-1]),
                                    initial_condition, t_eval=times)
            results[i] = sim_res.y.T

        return times, results

    def plot_mode_simulations(self, **parameter_overrides):

        times, results = self.simulate_modes(**parameter_overrides)

        fig, axes = plt.subplots(4, 2, sharex=True)

        for i, res in enumerate(results):
            axes[i, 0].plot(times, np.rad2deg(results[i, :, :2]))
            axes[i, 0].legend(['$' + lab + '$'
                               for lab in self.state_vars_latex[:2]])
            axes[i, 1].plot(times, np.rad2deg(results[i, :, 2:]))
            axes[i, 1].legend(['$' + lab + '$'
                               for lab in self.state_vars_latex[2:]])

        axes[3, 0].set_xlabel('Time [s]')
        axes[3, 1].set_xlabel('Time [s]')

        return axes
