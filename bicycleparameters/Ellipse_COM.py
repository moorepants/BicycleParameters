# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 16:07:58 2021

@author: Julie
"""
import numpy as np
from numpy import pi, sin, cos
# import plotly.graph_objects as go

def ell(x_center = 0, y_center = 0, ax1 = [1, 0], ax2 = [0, 1], a = 1, b = 1, N = 100):
    # x_center, y_center the coordinates of ellipse center
# ax1 ax2 two orthonormal vectors representing the ellipse axis directions
# a, b the ellipse parameters
    if np.linalg.norm(ax1) != 1 or np.linalg.norm(ax2) != 1:
        raise ValueError('ax1, ax2 must be unit vectors')
    if abs(np.dot(ax1, ax2)) > 1e-06:
        raise ValueError('ax1, ax2 must be orthogonal vectors')
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



def COM(R,x_center,y_center):
    t = np.linspace(0,0.5*pi, 100)
    xs = R*cos(t)
    ys = R*sin(t)   
    xc1 = xs + x_center
    yc1 = ys + y_center
    t2 = np.linspace(0.5*pi,pi, 100)
    xs2 = R*cos(t2)
    ys2 = R*sin(t2)
    xc2 = xs2 + x_center
    yc2 = ys2 + y_center
    t3 = np.linspace(pi,1.5*pi, 100)
    xs3 = R*cos(t3)
    ys3 = R*sin(t3)
    xc3 = xs3 + x_center
    yc3 = ys3 + y_center
    t4 = np.linspace(1.5*pi,2*pi, 100)
    xs4 = R*cos(t4)
    ys4 = R*sin(t4)
    xc4 = xs4 + x_center
    yc4 = ys4 + y_center
    # fig2=go.Figure()
    # fig2.add_trace(go.Scatter(x=[x_center,x_center+R],y=[y_center,y_center],mode='lines',line_color="black",showlegend = False))
    # fig2.add_trace(go.Scatter(x=[x_center,x_center],y=[y_center,y_center+R],mode='lines',line_color="black", showlegend = False))
    # fig2.add_trace(go.Scatter(x=xc1,y=yc1,mode='lines',line_color="black", showlegend = False,fill='tonexty'))
    # fig2.add_trace(go.Scatter(x=xc2,y=yc2,mode='lines',line_color="black", showlegend = False))
    # fig2.add_trace(go.Scatter(x=xc3,y=yc3,mode='lines',line_color="black", showlegend = False))
    # fig2.add_trace(go.Scatter(x=[x_center-R,x_center],y=[y_center,y_center],mode='lines',line_color="black",showlegend = False,fill='tonexty'))
    # fig2.add_trace(go.Scatter(x=[x_center,x_center],y=[y_center-R,y_center],mode='lines',line_color="black", showlegend = False))
    # fig2.add_trace(go.Scatter(x=xc4,y=yc4,mode='lines',line_color="black", showlegend = False))

    # return fig2
    return  xc1,xc2,xc3,xc4,yc1,yc2,yc3,yc4