#!/usr/bin/env python

from math import pi
from uncertainties import unumpy, umath, UFloat
import numpy as np

# local modules
from com import total_com

def combine_bike_rider(bicyclePar, riderPar):
    """
    Combines the inertia of the bicycle frame with the
    inertia of a rider.

    Parameters
    ----------
    bicyclePar : dictionary
        The benchmark parameter set of a bicycle.
    riderPar : dictionary
        The rider's mass, center of mass, and inertia expressed in the
        benchmark bicycle reference frame.

    Returns
    -------
    bicyclePar : dictionary
        The benchmark bicycle parameters with a rigid rider added to the
        bicycle frame.

    """

    # list the masses of the rider and bicycle
    masses = np.array([riderPar['mB'], bicyclePar['mB']])
    # list the centers of mass of the rider and bicycle
    coordinates = np.array([[riderPar['xB'], bicyclePar['xB']],
                            [riderPar['yB'], 0.],
                            [riderPar['zB'], bicyclePar['zB']]])
    # calculate the new mass and center of mass
    mT, cT = total_com(coordinates, masses)
    # get inertia tensors for the bicycle and rider
    IRider = part_inertia_tensor(riderPar, 'B')
    IBicycle = part_inertia_tensor(bicyclePar, 'B')
    # calculate the distance from the center of mass of each body to the
    # center of mass of the combined body
    dRider = np.array([riderPar['xB'] - cT[0],
                       riderPar['yB'] - cT[1],
                       riderPar['zB'] - cT[2]])
    dBicycle = np.array([bicyclePar['xB'] - cT[0],
                         0.,
                         bicyclePar['zB'] - cT[2]])
    # calculate the total inertia about the total body center of mass
    I = (parallel_axis(IRider, riderPar['mB'], dRider) +
         parallel_axis(IBicycle, bicyclePar['mB'], dBicycle))
    # assign new inertia back to bike
    bicyclePar['xB'] = cT[0]
    bicyclePar['zB'] = cT[2]
    bicyclePar['yB'] = 0.0
    bicyclePar['mB'] = mT
    bicyclePar['IBxx'] = I[0, 0]
    bicyclePar['IBxz'] = I[0, 2]
    bicyclePar['IByy'] = I[1, 1]
    bicyclePar['IBzz'] = I[2, 2]

    return bicyclePar

def compound_pendulum_inertia(m, g, l, T):
    '''Returns the moment of inertia for an object hung as a compound
    pendulum.

    Parameters
    ----------
    m : float
        Mass of the pendulum.
    g : float
        Acceration due to gravity.
    l : float
        Length of the pendulum.
    T : float
        The period of oscillation.

    Returns
    -------
    I : float
        Moment of interia of the pendulum.

    '''

    I = (T / 2. / pi)**2. * m * g * l - m * l**2.

    return I

def inertia_components(jay, beta):
    '''Returns the 2D orthogonal inertia tensor.

    When at least three moments of inertia and their axes orientations are
    known relative to a common inertial frame of a planar object, the orthoganl
    moments of inertia relative the frame are computed.

    Parameters
    ----------
    jay : ndarray, shape(n,)
        An array of at least three moments of inertia. (n >= 3)
    beta : ndarray, shape(n,)
        An array of orientation angles corresponding to the moments of inertia
        in jay.

    Returns
    -------
    eye : ndarray, shape(3,)
        Ixx, Ixz, Izz

    '''
    sb = unumpy.sin(beta)
    cb = unumpy.cos(beta)
    betaMat = unumpy.matrix(np.vstack((cb**2, -2 * sb * cb, sb**2)).T)
    eye = np.squeeze(np.asarray(np.dot(betaMat.I, jay)))
    return eye

def parallel_axis(Ic, m, d):
    '''Returns the moment of inertia of a body about a different point.

    Parameters
    ----------
    Ic : ndarray, shape(3,3)
        The moment of inertia about the center of mass of the body with respect
        to an orthogonal coordinate system.
    m : float
        The mass of the body.
    d : ndarray, shape(3,)
        The distances along the three ordinates that located the new point
        relative to the center of mass of the body.

    Returns
    -------
    I : ndarray, shape(3,3)
        The moment of inertia about of the body about a point located by d.

    '''
    a = d[0]
    b = d[1]
    c = d[2]
    dMat = np.zeros((3, 3), dtype=object)
    dMat[0] = np.array([b**2 + c**2, -a * b, -a * c])
    dMat[1] = np.array([-a * b, c**2 + a**2, -b * c])
    dMat[2] = np.array([-a * c, -b * c, a**2 + b**2])
    return Ic + m * dMat

def part_inertia_tensor(par, part):
    '''Returns an inertia tensor for a particular part for the benchmark
    parameter set.

    Parameters
    ----------
    par : dictionary
        Complete Benchmark parameter set.
    part : string
        Either 'B', 'H', 'F', 'R', 'G', 'S'

    Returns
    -------
    I : ndarray, shape(3,3)
        Inertia tensor for the part.

    '''
    if isinstance(par['mB'], UFloat):
        dtype=object
    else:
        dtype='float64'
    I = np.zeros((3, 3), dtype=dtype)
    # front or rear wheel
    if part == 'F' or part == 'R':
        axes = np.array([['xx', None, None],
                         [None, 'yy', None],
                         [None, None, 'xx']])
    # all other parts
    else:
        axes = np.array([['xx', None, 'xz'],
                         [None, 'yy', None],
                         ['xz', None, 'zz']])
    for i, row in enumerate(axes):
        for j, col in enumerate(row):
            if col != None:
                I[i, j] = par['I' + part + col]
    return I

def principal_axes(I):
    '''Returns the principal moments of inertia and the orientation.

    Parameters
    ----------
    I : ndarray, shape(3,3)
        An inertia tensor.

    Returns
    -------
    Ip : ndarray, shape(3,)
        The principal moments of inertia. This is sorted smallest to largest.
    C : ndarray, shape(3,3)
        The rotation matrix.

    '''
    Ip, C = np.linalg.eig(I)
    indices = np.argsort(Ip)
    Ip = Ip[indices]
    C = C.T[indices]
    return Ip, C

def rotate_inertia_tensor(I, angle):
    '''Returns inertia tensor rotated through angle. Only for 2D'''
    ca = umath.cos(angle)
    sa = umath.sin(angle)
    C    =  np.array([[ca, 0., -sa],
                      [0., 1., 0.],
                      [sa, 0., ca]])
    Irot =  np.dot(C, np.dot(I, C.T))
    return Irot

def tor_inertia(k, T):
    '''Calculate the moment of inertia for an ideal torsional pendulm

    Parameters
    ----------
    k: torsional stiffness
    T: period

    Returns
    -------
    I: moment of inertia

    '''

    I = k * T**2 / 4. / pi**2

    return I

def torsional_pendulum_stiffness(I, T):
    '''Calculate the stiffness of a torsional pendulum with a known moment of
    inertia.

    Parameters
    ----------
    I : moment of inertia
    T : period

    Returns
    -------
    k : stiffness

    '''
    k = 4. * I * pi**2 / T**2
    return k

def tube_inertia(l, m, ro, ri):
    '''Calculate the moment of inertia for a tube (or rod) where the x axis is
    aligned with the tube's axis.

    Parameters
    ----------
    l : float
        The length of the tube.
    m : float
        The mass of the tube.
    ro : float
        The outer radius of the tube.
    ri : float
        The inner radius of the tube. Set this to zero if it is a rod instead
        of a tube.

    Returns
    -------
    Ix : float
        Moment of inertia about tube axis.
    Iy, Iz : float
        Moment of inertia about normal axis.

    '''
    Ix = m / 2. * (ro**2 + ri**2)
    Iy = m / 12. * (3 * ro**2 + 3 * ri**2 + l**2)
    Iz = Iy
    return Ix, Iy, Iz
