# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 16:07:58 2021

@author: Julie
"""
import numpy as np
from numpy import pi, sin, cos
# import plotly.graph_objects as go

def ell(x_center, y_center, ax1, ax2, a, b, N = 100):
    # x_center, y_center the coordinates of ellipse center
# ax1 ax2 two orthonormal vectors representing the ellipse axis directions
# a, b the ellipse parameters
    # if np.linalg.norm(ax1) != 1 or np.linalg.norm(ax2) != 1:
    #     raise ValueError('ax1, ax2 must be unit vectors')
    # if abs(np.dot(ax1, ax2)) > 1e-06:
    #     raise ValueError('ax1, ax2 must be orthogonal vectors')
        
    # else:
        t = np.linspace(0, 2 * pi, N)
        #ellipse parameterization with respect to a system of axes of directions a1, a2
        xs = a * cos(t)
        ys = b * sin(t)
        #rotation matrix
        R = np.array([ax1, ax2]).T
        # coordinate of the ellipse points with respect to the system of axes[1, 0], [0, 1] with origin(0, 0)
        xp, yp = np.dot(R, [xs, ys])
        x = xp + x_center
        y = yp + y_center
        return x, y



