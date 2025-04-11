"""
Using Parameter Sets
====================

Parameter sets represent a set of constants associated with a multibody
dynamics model. These constants have a name and an associated floating point
value. This mapping from name to value is stored in a dictionary and then
passed to a :py:class:`~bicycleparameters.parameter_sets.ParameterSet` on
creation. The docstring of the parameter set shows what values must be defined
in the dictionary. This example will make use of the parameters associated with
the model defined in [Meijaard2007]_.
"""
from bicycleparameters import parameter_sets
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet

print(help(Meijaard2007ParameterSet))

# %%
# Below are a dictionary with parameters for the linear Carvallo-Whipple model
# with some realistic initial values.
par = {
    'IBxx': 11.3557360401,
    'IBxz': -1.96756380745,
    'IByy': 12.2177848012,
    'IBzz': 3.12354397008,
    'IFxx': 0.0904106601579,
    'IFyy': 0.149389340425,
    'IHxx': 0.253379594731,
    'IHxz': -0.0720452391817,
    'IHyy': 0.246138810935,
    'IHzz': 0.0955770796289,
    'IRxx': 0.0883819364527,
    'IRyy': 0.152467620286,
    'c': 0.0685808540382,
    'g': 9.81,
    'lam': 0.399680398707,
    'mB': 81.86,
    'mF': 2.02,
    'mH': 3.22,
    'mR': 3.11,
    'rF': 0.34352982332,
    'rR': 0.340958858855,
    'v': 1.0,
    'w': 1.121,
    'xB': 0.289099434117,
    'xH': 0.866949640247,
    'zB': -1.04029228321,
    'zH': -0.748236400835,
}

# %%
# The associated parameter set can then be created with the dictionary:
par_set = Meijaard2007ParameterSet(par, True)
par_set

# %%
# The dictionary of parameters is stored in the ``parameters`` attribute:
par_set.parameters

# %%
# The module :mod:`parameter_sets` includes different parameter sets and it may
# be possible to convert from one parameter set to another if the conversion is
# available.
print(parameter_sets.__all__)

# %%
# For example, this parameter set can be converted to the one for the linear
# Carvallo-Whipple model in [Moore2012]_:
par_set_moore = par_set.to_parameterization('Moore2012')
par_set_moore

# %%
# There is a unique label for each body embedded in the parameter variables,
# e.g. :math:`B` in :math:`I_{Bxx}`, and these are used in some methods below.
par_set.body_labels

# %%
# Many methods take one or more body labels as arguments. For example, the
# location of the combined mass center of the rear and front wheels can be
# found with:
par_set.mass_center_of('R', 'F')

# %%
# Or for all of the rigid bodies:
par_set.mass_center_of('B', 'H', 'F', 'R')

# %%
# The inertia tensor of a single body can be shown with:
par_set.form_inertia_tensor('B')

# %%
# Once the parameter set is available there are various methods that help you
# calculate and visualize the properties of this parameter set. This set
# describes the geometry, mass, and inertia of a bicycle. You can plot the
# geometry like so:
_ = par_set.plot_geometry()

# %%
# You can then add symbols representing the mass centers of the four bodies
# like so:
ax = par_set.plot_geometry()
_ = par_set.plot_mass_centers(ax=ax)

# %%
# You can then add symbols representing the radii of gyration of each rigid
# body like so:
ax = par_set.plot_geometry()
_ = par_set.plot_principal_radii_of_gyration(ax=ax)

# %%
# And finally, you can then add symbols representing uniformly dense ellipsoids
# with the same mass and inertia of each rigid body like so:
ax = par_set.plot_geometry()
_ = par_set.plot_principal_inertia_ellipsoids(ax=ax)

# %%
# All of the plot features can be shown with a single function call:
_ = par_set.plot_all()
