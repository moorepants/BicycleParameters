#!/usr/bin/env python

import numpy as np
from scipy.optimize import newton
from uncertainties import umath, ufloat, unumpy

def ab_matrix(M, C1, K0, K2, v, g):
    '''Calculate the A and B matrices for the Whipple bicycle model linearized
    about the upright configuration.

    Parameters
    ----------
    M : ndarray, shape(2,2)
        The mass matrix.
    C1 : ndarray, shape(2,2)
        The damping like matrix that is proportional to the speed, v.
    K0 : ndarray, shape(2,2)
        The stiffness matrix proportional to gravity, g.
    K2 : ndarray, shape(2,2)
        The stiffness matrix proportional to the speed squared, v**2.
    v : float
        Forward speed.
    g : float
        Acceleration due to gravity.

    Returns
    -------
    A : ndarray, shape(4,4)
        State matrix.
    B : ndarray, shape(4,2)
        Input matrix.

    The states are [roll rate,
                    steer rate,
                    roll angle,
                    steer angle]
    The inputs are [roll torque,
                    steer torque]

    '''

    a11 = -v * C1
    a12 = -(g * K0 + v**2 * K2)
    a21 = np.eye(2)
    a22 = np.zeros((2, 2))
    invM = (1. / (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) *
           np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]], dtype=M.dtype))
    A = np.vstack((np.dot(invM, np.hstack((a11, a12))),
                   np.hstack((a21, a22))))
    B = np.vstack((invM, np.zeros((2, 2))))

    return A, B

def benchmark_par_to_canonical(p):
    '''Returns the canonical matrices of the Whipple bicycle model linearized
    about the upright constant velocity configuration. It uses the parameter
    definitions from Meijaard et al. 2007.

    Parameters
    ----------
    p : dictionary
        A dictionary of the benchmark bicycle parameters. Make sure your units
        are correct, best to ue the benchmark paper's units!

    Returns
    -------
    M : ndarray, shape(2,2)
        The mass matrix.
    C1 : ndarray, shape(2,2)
        The damping like matrix that is proportional to the speed, v.
    K0 : ndarray, shape(2,2)
        The stiffness matrix proportional to gravity, g.
    K2 : ndarray, shape(2,2)
        The stiffness matrix proportional to the speed squared, v**2.

    Notes
    -----
    This function handles parameters with uncertanties.

    '''
    mT = p['mR'] + p['mB'] + p['mH'] + p['mF']
    xT = (p['xB'] * p['mB'] + p['xH'] * p['mH'] + p['w'] * p['mF']) / mT
    zT = (-p['rR'] * p['mR'] + p['zB'] * p['mB'] +
          p['zH'] * p['mH'] - p['rF'] * p['mF']) / mT

    ITxx = (p['IRxx'] + p['IBxx'] + p['IHxx'] + p['IFxx'] + p['mR'] *
            p['rR']**2 + p['mB'] * p['zB']**2 + p['mH'] * p['zH']**2 + p['mF']
            * p['rF']**2)
    ITxz = (p['IBxz'] + p['IHxz'] - p['mB'] * p['xB'] * p['zB'] -
            p['mH'] * p['xH'] * p['zH'] + p['mF'] * p['w'] * p['rF'])
    p['IRzz'] = p['IRxx']
    p['IFzz'] = p['IFxx']
    ITzz = (p['IRzz'] + p['IBzz'] + p['IHzz'] + p['IFzz'] +
            p['mB'] * p['xB']**2 + p['mH'] * p['xH']**2 + p['mF'] * p['w']**2)

    mA = p['mH'] + p['mF']
    xA = (p['xH'] * p['mH'] + p['w'] * p['mF']) / mA
    zA = (p['zH'] * p['mH'] - p['rF']* p['mF']) / mA

    IAxx = (p['IHxx'] + p['IFxx'] + p['mH'] * (p['zH'] - zA)**2 +
            p['mF'] * (p['rF'] + zA)**2)
    IAxz = (p['IHxz'] - p['mH'] * (p['xH'] - xA) * (p['zH'] - zA) + p['mF'] *
            (p['w'] - xA) * (p['rF'] + zA))
    IAzz = (p['IHzz'] + p['IFzz'] + p['mH'] * (p['xH'] - xA)**2 + p['mF'] *
            (p['w'] - xA)**2)
    uA = (xA - p['w'] - p['c']) * umath.cos(p['lam']) - zA * umath.sin(p['lam'])
    IAll = (mA * uA**2 + IAxx * umath.sin(p['lam'])**2 +
            2 * IAxz * umath.sin(p['lam']) * umath.cos(p['lam']) +
            IAzz * umath.cos(p['lam'])**2)
    IAlx = (-mA * uA * zA + IAxx * umath.sin(p['lam']) + IAxz *
            umath.cos(p['lam']))
    IAlz = (mA * uA * xA + IAxz * umath.sin(p['lam']) + IAzz *
            umath.cos(p['lam']))

    mu = p['c'] / p['w'] * umath.cos(p['lam'])

    SR = p['IRyy'] / p['rR']
    SF = p['IFyy'] / p['rF']
    ST = SR + SF
    SA = mA * uA + mu * mT * xT

    Mpp = ITxx
    Mpd = IAlx + mu * ITxz
    Mdp = Mpd
    Mdd = IAll + 2 * mu * IAlz + mu**2 * ITzz
    M = np.array([[Mpp, Mpd], [Mdp, Mdd]])

    K0pp = mT * zT # this value only reports to 13 digit precision it seems?
    K0pd = -SA
    K0dp = K0pd
    K0dd = -SA * umath.sin(p['lam'])
    K0 = np.array([[K0pp, K0pd], [K0dp, K0dd]])

    K2pp = 0.
    K2pd = (ST - mT * zT) / p['w'] * umath.cos(p['lam'])
    K2dp = 0.
    K2dd = (SA + SF * umath.sin(p['lam'])) / p['w'] * umath.cos(p['lam'])
    K2 = np.array([[K2pp, K2pd], [K2dp, K2dd]])

    C1pp = 0.
    C1pd = (mu*ST + SF*umath.cos(p['lam']) + ITxz / p['w'] *
            umath.cos(p['lam']) - mu*mT*zT)
    C1dp = -(mu * ST + SF * umath.cos(p['lam']))
    C1dd = (IAlz / p['w'] * umath.cos(p['lam']) + mu * (SA +
            ITzz / p['w'] * umath.cos(p['lam'])))
    C1 = np.array([[C1pp, C1pd], [C1dp, C1dd]])

    return M, C1, K0, K2

def lambda_from_abc(rF, rR, a, b, c):
    '''Returns the steer axis tilt, lamba, for the parameter set based on the
    offsets from the steer axis.

    Parameters
    ----------
    rF : float or ufloat
        Front wheel radius.
    rR : float or ufloat
        Rear wheel radius.
    a : float or ufloat
        The rear wheel offset. The minimum distance from the steer axis to the
        center of the rear wheel.
    b : float or ufloat
        The front wheel offset. The minimum distance from the steer axis to the
        center of the front wheel.
    c : float or ufloat
        The steer axis distance. The distance along the steer axis between the
        intersection of the front and rear wheel offset lines.

    '''
    def lam_equality(lam, rF, rR, a, b, c):
        return umath.sin(lam) - (rF - rR + c * umath.cos(lam)) / (a + b)
    guess = umath.atan(c / (a + b)) # guess based on equal wheel radii

    # The following assumes that the uncertainty calculated for the guess is
    # the same as the uncertainty for the true solution. This is not true! and
    # will surely breakdown the further the guess is away from the true
    # solution. There may be a way to calculate the correct uncertainity, but
    # that needs to be figured out. I guess I could use least squares and do it
    # the same way as get_period.

    args = (rF.nominal_value, rR.nominal_value, a.nominal_value,
            b.nominal_value, c.nominal_value)

    lam = newton(lam_equality, guess.nominal_value, args=args)
    return ufloat((lam, guess.std_dev()))

def sort_modes(evals, evecs):
    '''Sort eigenvalues and eigenvectors into weave, capsize, caster modes.

    Parameters
    ----------
    evals : ndarray, shape (n, 4)
        eigenvalues
    evecs : ndarray, shape (n, 4, 4)
        eigenvectors

    Returns
    -------
    weave['evals'] : ndarray, shape (n, 2)
        The eigen value pair associated with the weave mode.
    weave['evecs'] : ndarray, shape (n, 4, 2)
        The associated eigenvectors of the weave mode.
    capsize['evals'] : ndarray, shape (n,)
        The real eigenvalue associated with the capsize mode.
    capsize['evecs'] : ndarray, shape(n, 4, 1)
        The associated eigenvectors of the capsize mode.
    caster['evals'] : ndarray, shape (n,)
        The real eigenvalue associated with the caster mode.
    caster['evecs'] : ndarray, shape(n, 4, 1)
        The associated eigenvectors of the caster mode.

    Notes
    -----
    This only works on the standard bicycle eigenvalues, not necessarily on any
    general eigenvalues for the bike model (e.g. there isn't always a distinct
    weave, capsize and caster). Some type of check using the derivative of the
    curves could make it more robust.

    '''
    evalsorg = np.zeros_like(evals)
    evecsorg = np.zeros_like(evecs)
    # set the first row to be the same
    evalsorg[0] = evals[0]
    evecsorg[0] = evecs[0]
    # for each speed
    for i, speed in enumerate(evals):
        if i == evals.shape[0] - 1:
            break
        # for each current eigenvalue
        used = []
        for j, e in enumerate(speed):
            try:
                x, y = np.real(evalsorg[i, j].nominal_value), np.imag(evalsorg[i, j].nominal_value)
            except:
                x, y = np.real(evalsorg[i, j]), np.imag(evalsorg[i, j])
            # for each eigenvalue at the next speed
            dist = np.zeros(4)
            for k, eignext in enumerate(evals[i + 1]):
                try:
                    xn, yn = np.real(eignext.nominal_value), np.imag(eignext.nominal_value)
                except:
                    xn, yn = np.real(eignext), np.imag(eignext)
                # distance between points in the real/imag plane
                dist[k] = np.abs(((xn - x)**2 + (yn - y)**2)**0.5)
            if np.argmin(dist) in used:
                # set the already used indice higher
                dist[np.argmin(dist)] = np.max(dist) + 1.
            else:
                pass
            evalsorg[i + 1, j] = evals[i + 1, np.argmin(dist)]
            evecsorg[i + 1, :, j] = evecs[i + 1, :, np.argmin(dist)]
            # keep track of the indices we've used
            used.append(np.argmin(dist))
    weave = {'evals' : evalsorg[:, 2:], 'evecs' : evecsorg[:, :, 2:]}
    capsize = {'evals' : evalsorg[:, 1], 'evecs' : evecsorg[:, :, 1]}
    caster = {'evals' : evalsorg[:, 0], 'evecs' : evecsorg[:, :, 0]}
    return weave, capsize, caster

def trail(rF, lam, fo):
    '''Calculate the trail and mechanical trail.

    Parameters
    ----------
    rF : float
        The front wheel radius
    lam : float
        The steer axis tilt (pi/2 - headtube angle). The angle between the
        headtube and a vertical line.
    fo : float
        The fork offset

    Returns
    -------
    c: float
        Trail
    cm: float
        Mechanical Trail

    '''

    # trail
    c = (rF * umath.sin(lam) - fo) / umath.cos(lam)
    # mechanical trail
    cm = c * umath.cos(lam)
    return c, cm
