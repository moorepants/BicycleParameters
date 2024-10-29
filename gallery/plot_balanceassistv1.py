"""
Balance Assist V1
=================

.. figure:: https://objects-us-east-1.dream.io/mechmotum/balance-assist-bicycle-400x400.jpg
   :align: center

   Gazelle Grenoble/Arroyo E-Bike modified with a steering motor. Battery in
   the downtube and electronics box on the rear rack.

"""
import numpy as np
import bicycleparameters as bp
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
from bicycleparameters.models import Meijaard2007WithFeedbackModel

data_dir = "../data"

bicycle = bp.Bicycle("Balanceassistv1", pathToData=data_dir)
par = bp.io.remove_uncertainties(bicycle.parameters['Benchmark'])
par['v'] = 1.0
par_set = Meijaard2007ParameterSet(par, False)
par_set

# %%
par_set.plot_all()

# %%
model = Meijaard2007WithFeedbackModel(par_set)
v = np.linspace(0.0, 10.0, num=401)
ax = model.plot_eigenvalue_parts(v=v)
ax.set_ylim((-10.0, 10.0))

# %%
bicycle.add_rider('Jason', reCalc=True)
bicycle.plot_bicycle_geometry(inertiaEllipse=False)

# %%
par = bp.io.remove_uncertainties(bicycle.parameters['Benchmark'])
par['v'] = 1.0
par_set = Meijaard2007ParameterSet(par, True)
par_set.plot_all()

# %%
model = Meijaard2007WithFeedbackModel(par_set)
ax = model.plot_eigenvalue_parts(v=v)
ax.set_ylim((-10.0, 10.0))

# %%
# :math:`\kappa*(v_\textrm{max} - v)`
vmin, vmin_idx = 1.5, np.argmin(np.abs(v - 1.5))
vmax, vmax_idx = 4.7, np.argmin(np.abs(v - 4.7))
static_gain = 10.0
kphidots = -static_gain*(vmax - v)
kphidots[:vmin_idx] = -static_gain*(vmax - vmin)/vmin*v[:vmin_idx]
kphidots[vmax_idx:] = 0.0
model.plot_gains(v=v, kTdel_phid=kphidots)

# %%
ax = model.plot_eigenvalue_parts(v=v, kTdel_phid=kphidots)
ax.set_ylim((-10.0, 10.0))

# %%
times = np.linspace(0.0, 2.0, num=201)
x0 = np.array([0.0, 0.0, 0.5, 0.0])
kphidot = kphidots[np.argmin(np.abs(v - 3.0))]
ax = model.plot_simulation(times, x0, v=3.0)
ax = model.plot_simulation(times, x0, axes=ax, v=3.0, kTdel_phid=kphidot)
