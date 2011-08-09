#!/usr/bin/env python

import os
import numpy as np
from scipy.optimize import fmin
import yeadon

from io import remove_uncertainties
from inertia import combine_bike_rider
from geometry import calc_two_link_angles, vec_angle, vec_project

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
        human = rider_on_bike(bicyclePar, measuredPar,
                              pathToYeadon, pathToCFG, draw)
        # build a dictionary to store the inertial data
        riderPar = {'IBxx': human.Inertia[0, 0],
                    'IByy': human.Inertia[1, 1],
                    'IBzz': human.Inertia[2, 2],
                    'IBxz': human.Inertia[2, 0],
                    'mB': human.Mass,
                    'xB': human.COM[0][0],
                    'yB': human.COM[1][0],
                    'zB': human.COM[2][0]}
    except: #except if this fails
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
    H : yeadon.human
        Human object is returned with an updated configuration.
        The dictionary, taken from H.CFG, has the following key's values
        updated: ``CA1elevation``, ``CA1abduction``, ``A1A2flexion``,
        ``CB1elevation``, ``CB1abduction``, ``B1B2flexion``, ``PJ1elevation``,
        ``PJ1abduction``, ``J1J2flexion``, ``PK1elevation``, ``PK1abduction``,
        ``K1K2flexion``.

    Notes
    -----
    Requires that the bike object has a raw data text input file that contains
    the measurements necessary to situate a rider on the bike (i.e.
    ``<pathToData>/bicycles/<short name>/RawData/<short name>Measurements.txt``).

    .. _yeadon documentation : http://packages.python.org/yeadon


    """

    # create human using input measurements and configuration files
    H = yeadon.human(yeadonMeas, yeadonCFG)

    measuredPar = remove_uncertainties(measuredPar)
    benchmarkPar = remove_uncertainties(benchmarkPar)

    # for simplicity of code
    CFG = H.CFG
    # bottom bracket height
    hbb = measuredPar['hbb'] #.295
    # chain stay length
    Lcs = measuredPar['lcs'] #.46
    # rear wheel radius
    rR = benchmarkPar['rR'] #.342
    # front wheel radius
    rF = benchmarkPar['rF'] #.342
    # seat post length
    Lsp = measuredPar['lsp'] #.24
    # seat tube length
    Lst = measuredPar['lst'] #.53
    # seat tube angle
    lamst = measuredPar['lamst'] #68.5*np.pi/180
    # handlebar width
    whb = measuredPar['whb'] #43
    # distance from rear wheel hub to hand
    LhbR = measuredPar['LhbR'] #106
    # distance from front wheel hub to hand
    LhbF = measuredPar['LhbF'] #49
    # wheelbase
    w = benchmarkPar['w']

    # intermediate quantities
    # distance between wheel centers
    D = np.sqrt(w**2 + (rR - rF)**2)
    # projection into the plane of the bike
    dhbR = np.sqrt(LhbR**2 - (whb / 2)**2)
    dhbF = np.sqrt(LhbF**2 - (whb / 2)**2)
    # angle with vertex at rear hub, from horizontal "down" to front hub
    alpha = np.arcsin( (rR - rF) / D )
    # angle at rear hub of the dhbR-dhbF-D triangle (side-side-side)
    gamma = np.arccos( (dhbR**2 + D**2 - dhbF**2) / (2 * dhbR * D) )
    # position of bottom bracket center with respect to rear wheel contact
    # point
    pos_bb = np.array([[np.sqrt(Lcs**2 - (rR - hbb)**2)],
                       [0.],
                       [-hbb]])
    # vector from bottom bracket to seat
    vec_seat = -(Lst + Lsp) * np.array([[np.cos(lamst)],
                                        [0.],
                                        [np.sin(lamst)]])
    # position of seat with respect to rear wheel contact point
    pos_seat = pos_bb + vec_seat
    # vector (out of plane) from plane to right hand on the handlebars
    vec_hb_out  = np.array([[0.],
                            [whb / 2.],
                            [0.]])
    # vector (in plane) from rear wheel contact point to in-plane
    # location of hands
    vec_hb_in = np.array([[dhbR * np.cos(gamma - alpha)],
                          [0.],
                          [-rR - dhbR * np.sin(gamma - alpha)]])
    # position of right hand with respect to rear wheel contact point
    pos_handr = vec_hb_out + vec_hb_in
    # position of left hand with respect to rear wheel contact point
    pos_handl = -vec_hb_out + vec_hb_in

    # time to calculate the relevant quantities!
    # vector from seat to feet, ignoring out-of-plane distance
    vec_legs = -vec_seat
    # translation is done in bike's coordinate system
    H.translate_coord_sys(pos_seat)
    H.rotate_coord_sys((np.pi, 0., -np.pi /2.))
    # left foot
    pos_footl = pos_bb.copy()
    # set the y value at the same width as the hip
    pos_footl[1, 0] = H.J1.pos[1, 0]
    # right foot
    pos_footr = pos_bb.copy()
    # set the y value at the same width as the hip
    pos_footr[1, 0] = H.K1.pos[1, 0]

    # find the distance from the hip joint to the desired position of the foot
    DJ = np.linalg.norm( pos_footl - H.J1.pos)
    DK = np.linalg.norm( pos_footr - H.K1.pos)
    # find the distance from the should joint to the desired position of the
    # hands
    DA = np.linalg.norm( pos_handl - H.A1.pos)
    DB = np.linalg.norm( pos_handr - H.B1.pos)

    # distance from knees to heel level
    dj2 = np.linalg.norm( H.j[7].pos - H.J2.pos)
    dk2 = np.linalg.norm( H.k[7].pos - H.K2.pos)

    # distance from elbow to knuckle level
    da2 = np.linalg.norm( H.a[6].pos - H.A2.pos)
    db2 = np.linalg.norm( H.b[6].pos - H.B2.pos)

    # error-checking to make sure limbs are long enough for rider to sit
    # on the bike
    if (H.J1.length + dj2 < DJ):
        print "For the given measurements, the left leg is not " \
              "long enough. Left leg length is",H.J1.length+dj2, \
              "m, but distance from left hip joint to bottom bracket is", \
              DJ,"m."
        raise Exception()
    if (H.K1.length + dk2 < DK):
        print "For the given measurements, the right leg is not " \
              "long enough. Right leg length is",H.K1.length+dk2, \
              "m, but distance from right hip joint to bottom bracket is", \
              DK,"m."
        raise Exception()
    if (H.A1.length + da2 < DA):
        print "For the given configuration, the left arm is not " \
              "long enough. Left arm length is", H.A1.length + da2, \
              "m, but distance from shoulder to left hand is",DA,"m."
        raise Exception()
    if (H.B1.length + db2 < DB):
        print "For the given configuration, the right arm is not " \
              "long enough. Right arm length is",H.B1.length+db2, \
              "m, but distance from shoulder to right hand is",DB,"m."
        raise Exception()

    # joint angle time
    # legs first. torso cannot have twist
    # left leg
    tempangle, CFG['J1J2flexion'] =\
        calc_two_link_angles(H.J1.length, dj2, DJ)
    tempangle2 = vec_angle(np.array([[0,0,1]]).T, vec_legs)
    CFG['PJ1flexion'] = tempangle + tempangle2 + CFG['somersalt']
    # right leg
    tempangle,CFG['K1K2flexion'] =\
        calc_two_link_angles(H.K1.length, dk2, DK)
    CFG['PK1flexion'] = tempangle + tempangle2 + CFG['somersalt']

    # arms second. only somersalt can be specified, other torso
    # configuration variables must be zero

    def dist_hand_handle(angles, r_sh_hb, r_sh_h):
        """
        Returns the norm of the difference of the vector from the shoulder to
        the hand to the vector from the shoulder to the handlebar (i.e. the
        distance from the hand to the handlebar).

        Parameters
        ----------
        angles : array_like, shape(2,)
            The first angle is the elevation angle of the arm and the second is
            the abduction angle, both relative to the chest using euler 1-2-3
            angles.
        r_sh_hb : numpy.matrix, shape(3,1)
            The vector from the shoulder to the handlebar expressed in the
            chest reference frame.
        r_sh_h : numpy.matrix, shape(3,1)
            The vector from the shoulder to the hand (elbow is bent) expressed
            in the arm reference frame.

        Returns
        -------
        distance : float
            The distance from the handlebar point to the hand.

        """

        # these are euler rotation functions
        def x_rot(angle):
            sa = np.sin(angle)
            ca = np.cos(angle)
            Rx = np.matrix([[1., 0. , 0.],
                            [0., ca, sa],
                            [0., -sa, ca]])
            return Rx

        def y_rot(angle):
            sa = np.sin(angle)
            ca = np.cos(angle)
            Ry = np.matrix([[ca, 0. , -sa],
                            [0., 1., 0.],
                            [sa, 0., ca]])
            return Ry

        elevation = angles[0]
        abduction = angles[1]

        # create the rotation matrix of A (arm) in C (chest)
        R_A_C = y_rot(abduction) * x_rot(elevation)

        # express the vector from the shoulder to the hand in the C (chest)
        # refernce frame
        r_sh_h = R_A_C.T * r_sh_h

        return np.linalg.norm(r_sh_h - r_sh_hb)

    # left arm
    ##########
    tempangle, CFG['A1A2flexion'] =\
        calc_two_link_angles(H.A1.length, da2, DA)

    # this is the angle between the vector from the seat to the shoulder center
    # and the vector from the shoulder center to the handlebar center
    tempangle2 = vec_angle(vec_project(H.A1.pos - pos_seat, 1),
                            vec_project(pos_handl - H.A1.pos, 1))

    # the somersault angle plus the angle between the z unit vector and the
    # vector from the left shoulder to the left hand
    tempangle2 = CFG['somersalt'] + vec_angle(np.array([[0, 0, 1]]).T,
                                              pos_handl - H.A1.pos)

    # subtract the angle due to the arm not being straight
    CFG['CA1elevation'] = tempangle2 - tempangle

    # the angle between the vector from the shoulder to the handlebar and its
    # projection in the sagittal plane
    CFG['CA1abduction'] = vec_angle(pos_handl - H.A1.pos,
                                    vec_project(pos_handl - H.A1.pos, 1))

    # vector from the left shoulder to the left handlebar expressed in the
    # benchmark coordinates
    r_sh_hb = pos_handl - H.A1.pos
    # express r_sh_hb in the chest frame
    R_C_N = H.C.RotMat.T # transpose because Chris defined opposite my convention
    r_sh_hb = R_C_N * r_sh_hb
    # vector from the left shoulder to the hand (elbow bent) expressed in the
    # chest frame
    r_sh_h = np.mat([[0.],
                     [-da2 * np.sin(CFG['A1A2flexion'])],
                     [(-(H.A1.length + da2 *
                      np.cos(CFG['A1A2flexion'])))]])

    # chris defines his rotations relative to the arm coordinates but the
    # dist_hand_handle function is relative to the chest coordinates, thus the
    # negative guess
    guess = np.array([-CFG['CA1elevation'], -CFG['CA1abduction']])
    # home in on the exact solution
    elevation, abduction = fmin(dist_hand_handle, guess,
                                args=(r_sh_hb, r_sh_h), disp=False)
    # set the angles
    CFG['CA1elevation'], CFG['CA1abduction'] = -elevation, -abduction

    # right arm
    ###########
    tempangle, CFG['B1B2flexion'] =\
        calc_two_link_angles(H.B1.length, db2, DB)

    tempangle2 = vec_angle(vec_project(H.B1.pos - pos_seat, 1),
                           vec_project(pos_handr - H.B1.pos, 1))

    tempangle2 = CFG['somersalt'] + vec_angle(np.array([[0,0,1]]).T,
                                              pos_handr - H.B1.pos)

    CFG['CB1elevation'] = tempangle2 - tempangle
    CFG['CB1abduction'] = vec_angle(pos_handr - H.B1.pos,
                                    vec_project(pos_handr - H.B1.pos, 1))

    # vector from the left shoulder to the left handlebar expressed in the
    # benchmark coordinates
    r_sh_hb = pos_handr - H.B1.pos
    # express r_sh_hb in the chest frame
    R_C_N = H.C.RotMat.T # transpose because Chris defined opposite my convention
    r_sh_hb = R_C_N * r_sh_hb
    # vector from the left shoulder to the hand (elbow bent) expressed in the
    # chest frame
    r_sh_h = np.mat([[0.],
                     [-db2 * np.sin(CFG['B1B2flexion'])],
                     [(-(H.B1.length + db2 *
                      np.cos(CFG['B1B2flexion'])))]])

    # chris defines his rotations relative to the arm coordinates but the
    # dist_hand_handle function is relative to the chest coordinates, thus the
    # one negative guess
    guess = np.array([-CFG['CB1elevation'], CFG['CB1abduction']])
    # home in on the exact solution
    elevation, abduction = fmin(dist_hand_handle, guess,
                                args=(r_sh_hb, r_sh_h), disp=False)
    # set the angles
    CFG['CB1elevation'], CFG['CB1abduction'] = -elevation, abduction

    # assign configuration to human and check that the solution worked
    H.set_CFG_dict(CFG)
    if (np.round(H.j[7].pos, 3) != np.round(pos_footl, 3)).any():
        print "Left leg's actual position does not match its desired " \
              "position near the bike's bottom bracket. Left leg actual " \
              "position:\n", H.j[7].pos ,".\nLeft leg desired position:\n",\
              pos_footl, ".\nLeft leg base to end distance:", \
              np.linalg.norm(H.j[7].pos - H.J1.pos), ", Left leg D:", DJ
    if (np.round(H.k[7].pos, 3) != np.round(pos_footr, 3)).any():
        print "Right leg's actual position does not match its desired " \
              "position near the bike's bottom bracket. Right leg actual " \
              "position:\n", H.k[7].pos, ".\nRight leg desired position:\n",\
              pos_footr, ".\nRight leg base to end distance:", \
              np.linalg.norm(H.k[7].pos - H.K1.pos), ", Left leg D:", DK
    if (np.round(H.a[6].pos, 3) != np.round(pos_handl, 3)).any():
        print "Left arm's actual position does not match its desired " \
              "position on the bike's handlebar. Left arm actual " \
              "position:\n",H.A2.endpos,".\nLeft arm desired position:\n",\
              pos_handl, "\nLeft arm base to end distance:", \
              np.linalg.norm(H.A2.endpos - H.A1.pos),", Left arm D:", DA
    if (np.round(H.b[6].pos, 3) != np.round(pos_handr, 3)).any():
        print "Right arm's actual position does not match its desired " \
              "position on the bike's handrebar. Right arm actual " \
              "position:", H.B2.endpos ,".\nRight arm desired position:\n",\
              pos_handr, ".\nRight arm base to end distance:", \
              np.linalg.norm(H.B2.endpos - H.B1.pos), ", Right arm D:", DB

    # draw rider for fun, but possibly to check results aren't crazy
    if drawrider==True:
        H.draw_visual(forward=(0,-1,0),up=(0,0,-1))
        H.draw_vector('origin',pos_footl)
        H.draw_vector('origin',pos_footr)
        H.draw_vector('origin',pos_handl)
        H.draw_vector('origin',pos_handr)
        H.draw_vector('origin',H.A2.endpos)
        H.draw_vector('origin',H.A2.endpos,(0,0,1))
        H.draw_vector('origin',H.B2.endpos,(0,0,1))
    return H

