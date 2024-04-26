import numpy as np

from .com import total_com
from .inertia import parallel_axis


def _rotate_inertia_about_y(inertia, angle):
    """Returns inertia tensor rotated through an angle about the Y axis.

    Parameters
    ==========
    inertia : array_like, shape(3,3)
        An inertia tensor of a single rigid body defined relative to an XYZ
        Cartesian coordinate system.
    angle : float
        Angle in radians about the positive Y axis of which to rotate the
        coordinate system.

    Returns
    =======
    ndarray, shape(3,3)
        An inertia tensor of a single rigid body expressed relative to a
        rotated Cartesian coordinate system.

    """
    ca = np.cos(angle)
    sa = np.sin(angle)
    C = np.array([[ca, 0.0, -sa],
                  [0.0, 1., 0.0],
                  [sa, 0.0, ca]])
    return C @ inertia @ C.T


def _sum_central_inertias(m1, com1, I1, m2, com2, I2):
    # SUM_CENTRAL_INERTIAS - Returns the combined inertia of two bodies.
    #
    # Syntax : I = sum_central_inertias(m1, com1, I1, m2, com2, I2)
    #
    # Inputs:
    #   m1 - 1x1 double, mass of the first body
    #   com1 - 3x1 double, center of mass of the first body
    #   I1 - 3x3 double, central inertia tensor of the first body
    #   m2 - 1x1 double, mass of the second body
    #   com2 - 3x1 double, center of mass of the second body
    #   I2 - 3x3 double, central inertia tensor of the second body
    # Outputs:
    #   I - combined central inertia tensor of both bodies
    m, com = total_com(np.vstack([com1, com2]).T, [m1, m2])
    return parallel_axis(I1, m1, com-com1) + parallel_axis(I2, m2, com-com2)


def _convert_principal_to_benchmark(principal_par):
    # CONVERT_PRINCIPAL_TO_BENCHMARK - Returns a structure containing the
    # benchmark bicycle parameters as defined in Meijaard2007 which are
    # converted from the principal parameters as defined in MooreHubbard2019.
    #
    # Syntax: benchmark_par = convert_principal_to_benchmark(principal_par)
    #
    # Inputs:
    #   principal_par - Structure containing parameter names mapped to doubles.
    # Outputs:
    #   benchmark_par - Structure containing parameter names mapped to doubles.

    p = principal_par

    b = {}

    ###########################################################################
    # primary geometry, gravity, and speed
    ###########################################################################
    b["w"] = p["w"]
    b["c"] = p["c"]
    b["lam"] = p["lam"]
    b["g"] = p["g"]
    b["v"] = p["v"]

    ###########################################################################
    # rear frame [B]
    ###########################################################################
    coordinates = np.array([[p["xD"], p["xP"]],
                            [p["yD"], p["yP"]],
                            [p["zD"], p["zP"]]])
    masses = np.array([p["mD"], p["mP"]])
    b["mB"], (b["xB"], b["yB"], b["zB"]) = total_com(coordinates, masses)

    # symmetry is assumed about the XZ plane
    ID_principal = p["mD"]*np.diag([p["kDaa"]**2, p["kDyy"]**2, p["kDbb"]**2])
    IDxyz = _rotate_inertia_about_y(ID_principal, -p["alphaD"])

    # person
    IP_principal = p["mP"]*np.diag([p["kPaa"]**2, p["kPyy"]**2, p["kPbb"]**2])
    IPxyz = _rotate_inertia_about_y(IP_principal, -p["alphaP"])

    # combined rear frame and person
    comD = np.array([p["xD"], p["yD"], p["zD"]])
    comP = np.array([p["xP"], p["yP"], p["zP"]])
    IBxyz = _sum_central_inertias(p["mD"], comD, IDxyz, p["mP"], comP, IPxyz)
    b["IBxx"] = IBxyz[0, 0]
    b["IByy"] = IBxyz[1, 1]
    b["IBzz"] = IBxyz[2, 2]
    b["IBxz"] = IBxyz[0, 2]

    ###########################################################################
    # front frame [H]
    ###########################################################################
    b["mH"] = p["mH"]
    b["xH"] = p["xH"]
    b["yH"] = p["yH"]
    b["zH"] = p["zH"]

    # symmetry is assumed about the XZ plane
    IH123 = p["mH"]*np.diag([p["kHaa"]**2, p["kHyy"]**2, p["kHbb"]**2])
    IH = _rotate_inertia_about_y(IH123, -p["alphaH"])
    b["IHxx"] = IH[0, 0]
    b["IHyy"] = IH[1, 1]
    b["IHzz"] = IH[2, 2]
    b["IHxz"] = IH[0, 2]

    ###########################################################################
    # rear wheel [R]
    ###########################################################################
    b["rR"] = p["rR"]
    b["mR"] = p["mR"]
    # wheel is symmetric about XY, YZ, and XZ planes, thus no products of
    # inertia, wheel is ring or disc like, thus Ixx=Izz and Iyy > Ixx
    b["IRxx"] = p["mR"] * p["kRaa"]**2
    b["IRyy"] = p["mR"] * p["kRyy"]**2
    b["IRzz"] = p["mR"] * p["kRaa"]**2

    ###########################################################################
    # front wheel [F]
    ###########################################################################
    b["rF"] = p["rF"]
    b["mF"] = p["mF"]
    # wheel is symmetric about XY, YZ, and XZ planes, thus no prodcuts of
    # inertia, wheel is always ring or disc like, thus Ixx=Izz and Iyy > Ixx
    b["IFxx"] = p["mF"] * p["kFaa"]**2
    b["IFyy"] = p["mF"] * p["kFyy"]**2
    b["IFzz"] = p["mF"] * p["kFaa"]**2

    # return the benchmark parameters
    return b
