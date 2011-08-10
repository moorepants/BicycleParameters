#!/usr/bin/env python

from math import pi
import numpy as np
from uncertainties import unumpy, umath

from bicycle import trail, lambda_from_abc

def calculate_benchmark_geometry(mp, par):
    '''Returns the wheelbase, steer axis tilt and the trail.

    Parameters
    ----------
    mp : dictionary
        Dictionary with the measured parameters.
    par : dictionary
        Dictionary with the benchmark parameters.

    Returns
    -------
    par : dictionary
        par with the benchmark geometry added.

    '''
    # calculate the wheel radii
    par['rF'] = mp['dF'] / 2./ pi / mp['nF']
    par['rR'] = mp['dR'] / 2./ pi / mp['nR']

    # calculate the frame/fork fundamental geometry
    if 'w' in mp.keys(): # if there is a wheelbase
        # steer axis tilt in radians
        par['lam'] = pi / 180. * (90. - mp['gamma'])
        # wheelbase
        par['w'] = mp['w']
        # fork offset
        forkOffset = mp['f']
    else:
        h = (mp['h1'], mp['h2'], mp['h3'], mp['h4'], mp['h5'])
        d = (mp['d1'], mp['d2'], mp['d3'], mp['d4'], mp['d'])
        a, b, c = calculate_abc_geometry(h, d)
        par['lam'] = lambda_from_abc(par['rF'], par['rR'], a, b, c)
        par['w'] = (a + b) * umath.cos(par['lam']) + c * umath.sin(par['lam'])
        forkOffset = b

    # trail
    par['c'] = trail(par['rF'], par['lam'], forkOffset)[0]

    return par

def calculate_abc_geometry(h, d):
    '''Returns the perpendicular distance geometry for the bicycle from the raw
    measurements.

    Parameters
    ----------
    h : tuple
        Tuple containing the measured parameters h1-h5.
        (h1, h2, h3, h4, h5)
    d : tuple
        Tuple containing the measured parameters d1-d4 and d.
        (d1, d2, d3, d4, d)

    Returns
    -------
    a : ufloat or float
        The rear frame offset.
    b : ufloat or float
        The fork offset.
    c : ufloat or float
        The steer axis distance.

    '''
    # extract the values
    h1, h2, h3, h4, h5 = h
    d1, d2, d3, d4, d = d
    # get the perpendicular distances
    a = h1 + h2 - h3 + .5 * d1 - .5 * d2
    b = h4 - .5 * d3 - h5 + .5 * d4
    c = umath.sqrt(-(a - b)**2 + (d + .5 * (d2 + d3)))
    return a, b, c

def calculate_l1_l2(h6, h7, d5, d6, l):
    '''Returns the distance along (l2) and perpendicular (l1) to the steer axis from the
    front wheel center to the handlebar reference point.

    Parameters
    ----------
    h6 : float
        Distance from the table to the top of the front axle.
    h7 : float
        Distance from the table to the top of the handlebar reference circle.
    d5 : float
        Diameter of the front axle.
    d6 : float
        Diameter of the handlebar reference circle.
    l : float
        Outer distance from the front axle to the handlebar reference circle.

    Returns
    -------
    l1 : float
        The distance from the front wheel center to the handlebar reference
        center perpendicular to the steer axis. The positive sense is if the
        handlebar reference point is more forward than the front wheel center
        relative to the steer axis normal.
    l2 : float
       The distance from the front wheel center to the handlebar reference
       center parallel to the steer axis. The positive sense is if the
       handlebar reference point is above the front wheel center with reference
       to the steer axis.

    '''
    r5 = d5 / 2.
    r6 = d6 / 2.
    l1 = h7 - h6 + r5 - r6
    l0 = l - r5 - r6
    gamma = umath.asin(l1 / l0)
    l2 = l0 * umath.cos(gamma)
    return l1, l2

def calc_two_link_angles(L1, L2, D):
    '''Solves a simple case of the two-link revolute joint inverse
    kinematics problem. Both output angles are positive. The simple case
    is taht the end of the second link lies on the x-axis.

    Parameters
    ----------
    L1 : float
        Length of the first link.
    L2 : float
        Length of the second link.
    D : float
        Distance from the base of first link to the end of the second link.

    Returns
    -------
    theta1 : float
        (radians) Angle between x-axis and first link; always positive.
    theta2 : float
        (radians) Angle between first link and second link; always positive.

    '''

    theta1 = np.arccos( (L1**2 + D**2 - L2**2) / (2.0 * L1 * D) )
    theta2 = theta1 + np.arcsin( L1 / L2 * np.sin( theta1 ) )

    return theta1, theta2

def fwheel_to_handlebar_ref(lam, l1, l2):
    '''Returns the distance along the benchmark coordinates from the front
    wheel center to the handlebar reference center.

    Parameters
    ----------
    lam : float
        Steer axis tilt.
    l1, l2 : float
        The distance from the front wheel center to the handlebar refernce
        center perpendicular to and along the steer axis.

    Returns
    -------
    u1, u2 : float

    '''

    u1 = l2 * umath.sin(lam) - l1 * umath.cos(lam)
    u2 = u1 / umath.tan(lam) + l1 / umath.sin(lam)
    return u1, u2

def fundamental_geometry_plot_data(par):
    '''Returns the coordinates for line end points of the bicycle fundamental
    geometry.

    Parameters
    ----------
    par : dictionary
        Benchmark bicycle parameters.

    Returns
    -------
    x : ndarray
    z : ndarray

    '''
    d1 = np.cos(par['lam']) * (par['c'] + par['w'] -
                par['rR'] * np.tan(par['lam']))
    d3 = -np.cos(par['lam']) * (par['c'] - par['rF'] *
                np.tan(par['lam']))
    x = np.zeros(4, dtype=object)
    z = np.zeros(4, dtype=object)
    x[0] = 0.
    x[1] = d1 * np.cos(par['lam'])
    x[2] = par['w'] - d3 * np.cos(par['lam'])
    x[3] = par['w']
    z[0] = -par['rR']
    z[1] = -par['rR'] - d1 * np.sin(par['lam'])
    z[2] = -par['rF'] + d3 * np.sin(par['lam'])
    z[3] = -par['rF']

    return x, z

def point_to_line_distance(point, pointsOnLine):
    '''Returns the minimal distance from a point to a line in three
    dimensional space.

    Parameters
    ----------
    point : ndarray, shape(3,)
        The x, y, and z coordinates of a point.
    pointsOnLine : ndarray, shape(3,2)
        The x, y, and z coordinates of two points on a line. Rows are
        coordinates and columns are points.

    Returns
    -------
    distance : float
        The minimal distance from the line to the point.

    '''
    x1 = pointsOnLine[:, 0]
    x2 = pointsOnLine[:, 1]
    x0 = point

    def norm(v):
        return unumpy.sqrt(np.dot(v, v))

    distance = norm(np.cross((x0 - x1), (x0 - x2))) / norm(x2 - x1)

    return distance

def project_point_on_line(line, point):
    '''Returns point of projection.

    Parameters
    ----------
    line : tuple
        Slope and intercept of the line.
    point : tuple
        Location of the point.

    Returns
    -------
    newPoint : tuple
        The location of the projected point.

    '''
    m, b = line
    c , d = point
    x = (m * d + c - m * b) / (m**2. + 1.)
    y = (m**2. * d + m * c + b) / (m**2. + 1.)
    return x, y

def vec_angle(v1,v2):
    '''Returns the interior angle between two vectors using the dot product. Inputs do not need to be unit vectors.

    Parameters
    ----------
    v1 : np.array (3,1)
        input vector.
    v2 : np.array (3,1)
        input vector.

    Returns
    -------
    angle : float
        (radians) interior angle between v1 and v2.
    '''
    return np.arccos( float(np.dot(v1.T,v2)) / (
           np.linalg.norm(v1) * np.linalg.norm(v2) ) )

def vec_project(vec, direction):
    '''Vector projection into a plane, where the plane is defined by a
    normal vector.

    Parameters
    ----------
    vec : np.array(3,1)
        vector to be projected into a plane
    direction : int or np.array, shape(3,)
        If int, it is one of the three orthogonal directions, (0,1 or 2) of
        the input vector (essentially, that component of vec is set to zero).
        If np.array, can be in any direction (not necessarily a coordinate
        direction).

    Returns
    -------
    vec_out : np.array(3,1)
        Projected vector.

    '''
    vec = vec.flatten()
    if type(direction) == int:
        unitdir = np.zeros(3)
        unitdir[direction] = 1
    elif type(direction) == np.array:
        unitdir = direction / np.linalg.norm(direction)
    proj = vec - np.dot(vec, unitdir)
    return proj.reshape((3, 1))
