"""
Using Models
============

Parameter sets can be associated with a model and the model can be used to
compute and visualize properties of the model's dynamics.
"""
import numpy as np
from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
from bicycleparameters.models import Meijaard2007Model

np.set_printoptions(precision=3, suppress=True)

# %%
# Start with a parameter and construct a model with it:
par_set = Meijaard2007ParameterSet(meijaard2007_browser_jason, True)
model = Meijaard2007Model(par_set)

# %%
# Model Attributes
# ----------------
# Models have a set of ordered state variables:
model.state_vars

# %%
model.state_units

# %%
model.state_vars_latex

# %%
# Models have a set of ordered input variables:
model.input_vars

# %%
model.input_units

# %%
model.input_vars_latex

# %%
# State Space
# -----------
# All linear models have a method that generates the state and input matrices.
# The rows and columns match the order of the state and input variables.
A, B = model.form_state_space_matrices()
A

# %%
B

# %%
# Most all methods on models accept override values for the parameters. So if
# you want the state and input matrices with :math:`v=14,g=5.1` and :math:`w=2`
# pass them in as keyword arguments:
A, B = model.form_state_space_matrices(v=14.0, g=5.1, w=2.0)
A

# %%
B

# %%
# Eigenmodes
# ----------
# All linear models can produce its eigenvalues and eigenvectors:
evals, evecs = model.calc_eigen()
evals

# %%
evecs

# %%
# Parameters can be overridden as before, but the following line shows how you
# can pass in an iterable of values for any parameter override and get back an
# iterable of eigenvalues:
evals, evecs = model.calc_eigen(v=[2.0, 5.0])
evals

# %%
# The root locus with respect to any parameter, for example  here speed ``v``,
# can be plotted, which uses the iterable of eigenvalues shown above:
speeds = np.linspace(-10.0, 10.0, num=200)
_ = model.plot_eigenvalue_parts(v=speeds)

# %%
# There are several common customization options available for this plot:
speeds = np.linspace(0.0, 10.0, num=100)
ax = model.plot_eigenvalue_parts(v=speeds,
                                 colors=['C0', 'C0', 'C1', 'C2'],
                                 show_stable_regions=False,
                                 hide_zeros=True)
ax.set_ylim((-10.0, 10.0))
_ = ax.legend(['Weave Im', None, None, None,
               'Weave Re', None, 'Caster', 'Capsize'], ncol=4)

# %%
# You can choose any parameter in the dictionary to generate the root locus and
# also override other parameters.
wheelbases = np.linspace(0.2, 5.0, num=50)
_ = model.plot_eigenvalue_parts(v=6.0, w=wheelbases)

# %%
# It is also possible to pass iterables for multiple parameters as long as the
# length is the same for all. Only the scale for the first will be shown.
trails = np.linspace(-0.2, 0.2, num=50)
wheelbases = np.linspace(0.2, 5.0, num=50)
_ = model.plot_eigenvalue_parts(c=trails, w=wheelbases)

# %%
# The eigenvector components can be created for each mode and for a series of
# parameter values:
_ = model.plot_eigenvectors(v=[1.0, 3.0, 5.0, 7.0])

# %%
# The eigenmodes can be simulated for specific parameter values:
times = np.linspace(0.0, 5.0, num=100)
_ = model.plot_mode_simulations(times, v=6.0)

# %%
# Simulation
# ----------
# A general simulation from initial conditions can also be run:
x0 = np.deg2rad([5.0, -3.0, 0.0, 0.0])
_ = model.plot_simulation(times, x0, v=6.0)

# %%
# Inputs can be applied in the simulation, for example a simple positive
# feedback derivative controller on roll shows that the low speed bicycle can
# be stabilized:
x0 = np.deg2rad([5.0, -3.0, 0.0, 0.0])
_ = model.plot_simulation(times, x0,
                          input_func=lambda t, x: np.array([0.0, 50.0*x[2]]),
                          v=2.0)
