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
from bicycleparameters.models import Meijaard2007Model

data_dir = "../data"

bicycle = bp.Bicycle("Balanceassistv1", pathToData=data_dir)
par = bp.io.remove_uncertainties(bicycle.parameters['Benchmark'])
par['v'] = 1.0
par_set = Meijaard2007ParameterSet(par, False)
par_set.plot_all()

# %%
model = Meijaard2007Model(par_set)
v = np.linspace(0.0, 10.0, num=401)
model.plot_eigenvalue_parts(v=v)

# %%
bicycle.add_rider('Jason', reCalc=True)
bicycle.plot_bicycle_geometry(inertiaEllipse=False)

# %%
par = bp.io.remove_uncertainties(bicycle.parameters['Benchmark'])
par['v'] = 1.0
par_set = Meijaard2007ParameterSet(par, True)
model = Meijaard2007Model(par_set)
model.plot_eigenvalue_parts(v=v)
