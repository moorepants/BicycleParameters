#!/usr/bin/env python

import os
import numpy as np
from numpy import sin, cos, sqrt
from scipy.optimize import fsolve
import yeadon

from .io import remove_uncertainties
from .inertia import combine_bike_rider


def yeadon_vec_to_bicycle_vec(vector, measured_bicycle_par,
                              benchmark_bicycle_par):
    """
    Parameters
    ----------
    vector : np.matrix, shape(3, 1)
        A vector from the Yeadon origin to a point expressed in the Yeadon
        reference frame.
    measured_bicycle_par : dictionary
        The raw bicycle measurements.
    benchmark_bicycle_par : dictionary
        The Meijaard 2007 et. al parameters for this bicycle.

    Returns
    -------
    vector_wrt_bike : np.matrix, shape(3, 1)
        The vector from the bicycle origin to the same point above expressed
        in the bicycle reference frame.

    """

    # This is the rotation matrix that relates Yeadon's reference frame
    # to the bicycle reference frame.
    # vector_expressed_in_bike = rot_mat * vector_expressed_in_yeadon)
    rot_mat = np.matrix([[0.0, -1.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0]])

    # The relevant bicycle parameters:
    measuredPar = remove_uncertainties(measured_bicycle_par)
    benchmarkPar = remove_uncertainties(benchmark_bicycle_par)
    # bottom bracket height
    hbb = measuredPar['hbb']
    # chain stay length
    lcs = measuredPar['lcs']
    # rear wheel radius
    rR = benchmarkPar['rR']
    # seat post length
    lsp = measuredPar['lsp']
    # seat tube length
    lst = measuredPar['lst']
    # seat tube angle
    lambdast = measuredPar['lamst']

    # bicycle origin to yeadon origin expressed in bicycle frame
    yeadon_origin_in_bike_frame = \
        np.matrix([[np.sqrt(lcs**2 - (-hbb + rR)**2) + (-lsp - lst) * np.cos(lambdast)],  # bx
                   [0.0],
                   [-hbb + (-lsp - lst) * np.sin(lambdast)]])  # bz

    vector_wrt_bike =  yeadon_origin_in_bike_frame + rot_mat * vector

    return vector_wrt_bike


def configure_rider(pathToRider, bicycle, bicyclePar, measuredPar, draw):
    """
    Returns the rider parameters, bicycle paramaters with a rider and a
    human object that is configured to sit on the bicycle.

    Parameters
    ----------
    pathToRider : string
        Path to the rider's data folder.
    bicycle : string
        The short name of the bicycle.
    bicyclePar : dictionary
        Contains the benchmark bicycle parameters for a bicycle.
    measuredPar : dictionary
        Contains the measured values of the bicycle.
    draw : boolean, optional
        If true, visual python will be used to draw a three dimensional
        image of the rider.

    Returns
    -------
    riderpar : dictionary
        The inertial parameters of the rider with reference to the
        benchmark coordinate system.
    human : yeadon.human
        The human object that represents the rider seated on the
        bicycle.
    bicycleRiderPar : dictionary
        The benchmark parameters of the bicycle with the rider added to
        the rear frame.

    """
    try:
        # get the rider name
        rider = os.path.split(pathToRider)[1]
        # get the paths to the yeadon data files
        pathToYeadon = os.path.join(pathToRider, 'RawData',
                                    rider + 'YeadonMeas.txt')
        pathToCFG = os.path.join(pathToRider, 'RawData',
                                 rider + bicycle + 'YeadonCFG.txt')
        # generate the human that has been configured to sit on the bicycle
        # the human's inertial parameters are expressed in the Yeadon
        # reference frame about the Yeadon origin.
        human = rider_on_bike(bicyclePar, measuredPar,
                              pathToYeadon, pathToCFG, draw)

        # This is the rotation matrix that relates Yeadon's reference frame
        # to the bicycle reference frame.
        rot_mat = np.array([[0.0, -1.0, 0.0],
                            [-1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0]])

        # This is the human's inertia expressed in the bicycle reference
        # frame about the human's center of mass.
        human_inertia_in_bike_frame = \
            human.inertia_transformed(rotmat=rot_mat)

        human_com_in_bike_frame = \
            yeadon_vec_to_bicycle_vec(human.center_of_mass, measuredPar,
                                      bicyclePar)

        # build a dictionary to store the inertial data
        riderPar = {'IBxx': human_inertia_in_bike_frame[0, 0],
                    'IByy': human_inertia_in_bike_frame[1, 1],
                    'IBzz': human_inertia_in_bike_frame[2, 2],
                    'IBxz': human_inertia_in_bike_frame[2, 0],
                    'mB': human.mass,
                    'xB': human_com_in_bike_frame[0, 0],
                    'yB': human_com_in_bike_frame[1, 0],
                    'zB': human_com_in_bike_frame[2, 0]}
    except:  # except if this fails
        # no rider was added
        print('Calculations in yeadon failed. No rider added.')
        # raise the error that caused things to fail
        raise
    else:
        bicycleRiderPar = combine_bike_rider(bicyclePar, riderPar)
        return riderPar, human, bicycleRiderPar


def rider_on_bike(benchmarkPar, measuredPar, yeadonMeas, yeadonCFG,
                  drawrider):
    """
    Returns a yeadon human configured to sit on a bicycle.

    Parameters
    ----------
    benchmarkPar : dictionary
        A dictionary containing the benchmark bicycle parameters.
    measuredPar : dictionary
        A dictionary containing the raw geometric measurements of the bicycle.
    yeadonMeas : str
        Path to a text file that holds the 95 yeadon measurements. See
        `yeadon documentation`_.
    yeadonCFG : str
        Path to a text file that holds configuration variables. See `yeadon
        documentation`_. As of now, only 'somersalt' angle can be set as an
        input. The remaining variables are either zero or calculated in this
        method.
    drawrider : bool
        Switch to draw the rider, with vectors pointing to the desired
        position of the hands and feet of the rider (at the handles and
        bottom bracket). Requires python-visual.

    Returns
    -------
    human : yeadon.Human
        Human object is returned with an updated configuration.
        The dictionary, taken from H.CFG, has the following key's values
        updated:

            'PJ1extension'
            'J1J2flexion'
            'CA1extension'
            'CA1adduction'
            'CA1rotation'
            'A1A2extension'
            'somersault'
            'PK1extension'
            'K1K2flexion'
            'CB1extension'
            'CB1abduction'
            'CB1rotation'
            'B1B2extension'

    Notes
    -----
    Requires that the bike object has a raw data text input file that contains
    the measurements necessary to situate a rider on the bike (i.e.
    ``<pathToData>/bicycles/<short name>/RawData/<short name>Measurements.txt``).

    .. _yeadon documentation : http://packages.python.org/yeadon


    """

    # create human using input measurements and configuration files
    human = yeadon.Human(yeadonMeas, yeadonCFG)

    # The relevant human measurments:
    L_j3L = human.meas['Lj3L']
    L_j5L = human.meas['Lj5L']
    L_j6L = human.meas['Lj6L']
    L_s4L = human.meas['Ls4L']
    L_s4w = human.meas['Ls4w']
    L_a2L = human.meas['La2L']
    L_a4L = human.meas['La4L']
    L_a5L = human.meas['La5L']
    somersault = human.CFG['somersault']

    # The relevant bicycle parameters:
    measuredPar = remove_uncertainties(measuredPar)
    benchmarkPar = remove_uncertainties(benchmarkPar)
    # bottom bracket height
    h_bb = measuredPar['hbb']
    # chain stay length
    l_cs = measuredPar['lcs']
    # rear wheel radius
    r_R = benchmarkPar['rR']
    # front wheel radius
    r_F = benchmarkPar['rF']
    # seat post length
    l_sp = measuredPar['lsp']
    # seat tube length
    l_st = measuredPar['lst']
    # seat tube angle
    lambda_st = measuredPar['lamst']
    # handlebar width
    w_hb = measuredPar['whb']
    # distance from rear wheel hub to hand
    L_hbR = measuredPar['LhbR']
    # distance from front wheel hub to hand
    L_hbF = measuredPar['LhbF']
    # wheelbase
    w = benchmarkPar['w']

    def zero(unknowns):
        """For the derivation of these equations see:

           http://nbviewer.ipython.org/github/chrisdembia/yeadon/blob/v1.2.0/examples/bicyclerider/bicycle_example.ipynb
        """

        PJ1extension = unknowns[0]
        J1J2flexion = unknowns[1]
        CA1extension = unknowns[2]
        CA1adduction = unknowns[3]
        CA1rotation = unknowns[4]
        A1A2extension = unknowns[5]
        alpha_y = unknowns[6]
        alpha_z = unknowns[7]
        beta_y = unknowns[8]
        beta_z = unknowns[9]

        phi_J1 = PJ1extension
        phi_J2 = J1J2flexion
        phi_A1 = CA1extension
        theta_A1 = CA1adduction
        psi_A = CA1rotation
        phi_A2 = A1A2extension

        phi_P = somersault

        zero = np.zeros(10)

        zero[0] = (L_j3L*(-sin(phi_J1)*cos(phi_P) - sin(phi_P)*cos(phi_J1))
                   + (-l_sp - l_st)*cos(lambda_st) + (-(-sin(phi_J1)*
                   sin(phi_P) + cos(phi_J1)*cos(phi_P))*sin(phi_J2) +
                   (-sin(phi_J1)*cos(phi_P) - sin(phi_P)*cos(phi_J1))*
                   cos(phi_J2))*(-L_j3L + L_j5L + L_j6L))

        zero[1] = (L_j3L*(-sin(phi_J1)*sin(phi_P) + cos(phi_J1)*cos(phi_P))
                   + (-l_sp - l_st)*sin(lambda_st) + ((-sin(phi_J1)*
                   sin(phi_P) + cos(phi_J1)*cos(phi_P))*cos(phi_J2) -
                   (sin(phi_J1)*cos(phi_P) + sin(phi_P)*cos(phi_J1))*
                   sin(phi_J2))*(-L_j3L + L_j5L + L_j6L))

        zero[2] = -L_hbF + sqrt(alpha_y**2 + alpha_z**2 + 0.25*w_hb**2)

        zero[3] = -L_hbR + sqrt(beta_y**2 + beta_z**2 + 0.25*w_hb**2)

        zero[4] = alpha_y - beta_y - w

        zero[5] = alpha_z - beta_z + r_F - r_R

        zero[6] = (-L_a2L*sin(theta_A1) + L_s4w/2 - 0.5*w_hb + (sin(phi_A2)*
                   sin(psi_A)*cos(theta_A1) + sin(theta_A1)*cos(phi_A2))*
                   (L_a2L - L_a4L - L_a5L))

        zero[7] = (-L_a2L*(-sin(phi_A1)*cos(phi_P)*cos(theta_A1) -
                   sin(phi_P)*cos(phi_A1)*cos(theta_A1)) - L_s4L*sin(phi_P)
                   - beta_y - sqrt(l_cs**2 - (-h_bb + r_R)**2) - (-l_sp -
                   l_st)*cos(lambda_st) + (-(-(sin(phi_A1)*cos(psi_A) +
                   sin(psi_A)*sin(theta_A1)*cos(phi_A1))*sin(phi_P) +
                   (-sin(phi_A1)*sin(psi_A)*sin(theta_A1) + cos(phi_A1)*
                   cos(psi_A))*cos(phi_P))*sin(phi_A2) + (-sin(phi_A1)*
                   cos(phi_P)*cos(theta_A1) - sin(phi_P)*cos(phi_A1)*
                   cos(theta_A1))*cos(phi_A2))*(L_a2L - L_a4L - L_a5L))

        zero[8] = (-L_a2L*(-sin(phi_A1)*sin(phi_P)*cos(theta_A1) +
                   cos(phi_A1)*cos(phi_P)*cos(theta_A1)) + L_s4L*cos(phi_P)
                   - beta_z + h_bb - r_R - (-l_sp - l_st)*sin(lambda_st) +
                   (-((sin(phi_A1)*cos(psi_A) + sin(psi_A)*sin(theta_A1)*
                   cos(phi_A1))*cos(phi_P) + (-sin(phi_A1)*sin(psi_A)*
                   sin(theta_A1) + cos(phi_A1)*cos(psi_A))*sin(phi_P))*
                   sin(phi_A2) + (-sin(phi_A1)*sin(phi_P)*cos(theta_A1) +
                   cos(phi_A1)*cos(phi_P)*cos(theta_A1))*cos(phi_A2))*(L_a2L
                   - L_a4L - L_a5L))

        zero[9] = ((sin(phi_A1)*sin(psi_A) - sin(theta_A1)*cos(phi_A1)*
                    cos(psi_A))*cos(phi_P) + (sin(phi_A1)*sin(theta_A1)*
                    cos(psi_A) + sin(psi_A)*cos(phi_A1))*sin(phi_P))

        return zero

    g_PJ1extension = -np.deg2rad(90.0)
    g_J1J2flexion = np.deg2rad(75.0)
    g_CA1extension = -np.deg2rad(15.0)
    g_CA1adduction = np.deg2rad(2.0)
    g_CA1rotation = np.deg2rad(2.0)
    g_A1A2extension = -np.deg2rad(40.0)
    g_alpha_y = L_hbF * np.cos(np.deg2rad(45.0))
    g_alpha_z = L_hbF * np.sin(np.deg2rad(45.0))
    g_beta_y = -L_hbR * np.cos(np.deg2rad(30.0))
    g_beta_z = L_hbR * np.sin(np.deg2rad(30.0))

    guess = [g_PJ1extension, g_J1J2flexion, g_CA1extension, g_CA1adduction,
             g_CA1rotation, g_A1A2extension, g_alpha_y, g_alpha_z, g_beta_y,
             g_beta_z]

    solution = fsolve(zero, guess)

    cfg_dict = human.CFG.copy()
    cfg_dict['PJ1extension'] = solution[0]
    cfg_dict['J1J2flexion'] = solution[1]
    cfg_dict['CA1extension'] = solution[2]
    cfg_dict['CA1adduction'] = solution[3]
    cfg_dict['CA1rotation'] = solution[4]
    cfg_dict['A1A2extension'] = solution[5]
    cfg_dict['somersault'] = somersault
    cfg_dict['PK1extension'] = cfg_dict['PJ1extension']
    cfg_dict['K1K2flexion'] = cfg_dict['J1J2flexion']
    cfg_dict['CB1extension'] = cfg_dict['CA1extension']
    cfg_dict['CB1abduction'] = -cfg_dict['CA1adduction']
    cfg_dict['CB1rotation'] = -cfg_dict['CA1rotation']
    cfg_dict['B1B2extension'] = cfg_dict['A1A2extension']

    # assign configuration to human and check that the solution worked
    human.set_CFG_dict(cfg_dict)

    # draw rider for fun, but possibly to check results aren't crazy
    if drawrider:
        human.draw()

    return human
