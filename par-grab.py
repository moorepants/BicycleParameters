import os
import bicycleparameters as bp
import pandas as pd
import numpy as np 
from bicycleparameters import parameter_sets as ps

bike = bp.Bicycle('Benchmark', pathToData=os.getcwd()+'\\data')
par = bike.parameters['Benchmark']
parPure = bp.io.remove_uncertainties(par)
print(parPure) # prints a dictionary of variable:value pairs

'''
pset = ps.ParameterSet(par) # needs par_string(?)

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
'''