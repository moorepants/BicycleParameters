#!/usr/bin/env python

from math import pi
import numpy as np
from uncertainties import umath, unumpy

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

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def center_of_mass(slopes, intercepts):
    '''Returns the center of mass relative to the slopes and intercepts
    coordinate system.

    Parameters
    ----------
    slopes : ndarray, shape(n,)
        The slope of every line used to calculate the center of mass.
    intercepts : ndarray, shape(n,)
        The intercept of every line used to calculate the center of mass.

    Returns
    -------
    x : float
        The abscissa of the center of mass.
    y : float
        The ordinate of the center of mass.

    '''
    num = range(len(slopes))
    allComb = cartesian((num, num))
    comb = []
    # remove doubles
    for row in allComb:
        if row[0] != row[1]:
            comb.append(row)
    comb = np.array(comb)

    # initialize the matrix to store the line intersections
    lineX = np.zeros((len(comb), 2), dtype='object')
    # for each line intersection...
    for j, row in enumerate(comb):
        sl = np.array([slopes[row[0]], slopes[row[1]]])
        a = unumpy.matrix(np.vstack((-sl, np.ones((2)))).T)
        b = np.array([intercepts[row[0]], intercepts[row[1]]])
        lineX[j] = np.dot(a.I, b)
    com = np.mean(lineX, axis=0)

    return com[0], com[1]

def com_line(alpha, a, par, part, l1, l2):
    '''Returns the slope and intercept for the line that passes through the
    part's center of mass with reference to the benchmark bicycle coordinate
    system.

    Parameters
    ----------
    alpha : float
        The angle the head tube makes with the horizontal. When looking at the
        bicycle from the right side this is the angle between a vector point
        out upwards along the steer axis and the earth horizontal with the
        positve direction pointing from the left to the right. If the bike is
        in its normal configuration this would be 90 degrees plus the steer
        axis tilt (lambda).
    a : float
        The distance from the pendulum axis to a reference point on the part,
        typically the wheel centers. This is positive if the point falls to the
        left of the axis and negative otherwise.
    par : dictionary
        Benchmark parameters. Must include lam, rR, rF, w
    part : string
        The subscript denoting which part this refers to.
    l1, l2 : floats
        The location of the handlebar reference point relative to the front
        wheel center when the fork is split. This is measured perpendicular to
        and along the steer axis, respectively.

    Returns
    -------
    m : float
        The slope of the line in the benchmark coordinate system.
    b : float
        The z intercept in the benchmark coordinate system.

    '''

    # beta is the angle between the x bike frame and the x pendulum frame, rotation
    # about positive y
    beta = par['lam'] - alpha * pi / 180

    # calculate the slope of the center of mass line
    m = -umath.tan(beta)

    # calculate the z intercept
    # this the bicycle frame
    if part == 'B':
        b = -a / umath.cos(beta) - par['rR']
    # this is the fork (without handlebar) or the fork and handlebar combined
    elif part == 'S' or part == 'H':
        b = -a / umath.cos(beta) - par['rF'] + par['w'] * umath.tan(beta)
    # this is the handlebar (without fork)
    elif part == 'G':
        u1, u2 = fwheel_to_handlebar_ref(par['lam'], l1, l2)
        b = -a / umath.cos(beta) - (par['rF'] + u2) + (par['w'] - u1) * umath.tan(beta)
    else:
        print part, "doesn't exist"
        raise KeyError

    return m, b, beta

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

def part_com_lines(mp, par, forkIsSplit):
    '''Returns the slopes and intercepts for all of the center of mass lines
    for each part.

    Parameters
    ----------
    mp : dictionary
        Dictionary with the measured parameters.

    Returns
    -------
    slopes : dictionary
        Contains a list of slopes for each part.
    intercepts : dictionary
        Contains a list of intercepts for each part.

    The slopes and intercepts lists are in order with respect to each other and
    the keyword is either 'B', 'H', 'G' or 'S' for Frame, Handlebar/Fork,
    Handlerbar, and Fork respectively.

    '''
    # find the slope and intercept for pendulum axis
    if forkIsSplit:
        l1, l2 = calculate_l1_l2(mp['h6'], mp['h7'], mp['d5'],
                                 mp['d6'], mp['l'])
        slopes = {'B':[], 'G':[], 'S':[]}
        intercepts = {'B':[], 'G':[], 'S':[]}
        betas = {'B':[], 'G':[], 'S':[]}
    else:
        l1, l2 = 0., 0.
        slopes = {'B':[], 'H':[]}
        intercepts = {'B':[], 'H':[]}
        betas = {'B':[], 'H':[]}

    # get the alpha keys and put them in order
    listOfAlphaKeys = [x for x in mp.keys() if x.startswith('alpha')]
    listOfAlphaKeys.sort()

    # caluclate m, b and beta for each orientation
    for key in listOfAlphaKeys:
        alpha = mp[key]
        a = mp['a' + key[5:]]
        part = key[5]
        m, b, beta = com_line(alpha, a, par, part, l1, l2)
        slopes[part].append(m)
        intercepts[part].append(b)
        betas[part].append(beta)

    return slopes, intercepts, betas
