import numpy as np
from numpy import pi, sin, cos, sqrt


def IRTNN(data):
    u0 = data["u0"]
    u1 = data["u1"]
    u2 = data["u2"]
    u12 = data["u12"]
    u11 = data["u11"]
    u22 = data["u22"]

    D_C3 = np.array(
        [
            [1, 0, 0],
            [0, cos(-2 * pi / 3), -sin(-2 * pi / 3)],
            [0, sin(-2 * pi / 3), cos(-2 * pi / 3)],
        ]
    )

    D_2C3 = np.array(
        [
            [1, 0, 0],
            [0, cos(-4 * pi / 3), -sin(-4 * pi / 3)],
            [0, sin(-4 * pi / 3), cos(-4 * pi / 3)],
        ]
    )

    D_S = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]
    )

    D_S1 = np.array(
        [
            [1, 0, 0],
            [0, 1 / 2, -sqrt(3) / 2],
            [0, -sqrt(3) / 2, -1 / 2],
        ]
    )

    D_S2 = np.array(
        [
            [1, 0, 0],
            [0, 1 / 2, sqrt(3) / 2],
            [0, sqrt(3) / 2, -1 / 2],
        ]
    )

    E_R7 = np.array(
        [
            [u0, u1, u2],
            [-u1, u11, u12],
            [u2, -u12, u22],
        ]
    )

    E_R8 = D_S1 @ E_R7 @ D_S1.T
    E_R9 = D_C3 @ E_R7 @ D_C3.T
    E_R10 = D_S @ E_R7 @ D_S.T
    E_R11 = D_2C3 @ E_R7 @ D_2C3.T
    E_R12 = D_S2 @ E_R7 @ D_S2.T

    return E_R7, E_R8, E_R9, E_R10, E_R11, E_R12


def IRNN(data):
    r0 = data["r0"]
    r1 = data["r1"]
    r2 = data["r2"]
    r12 = data["r12"]
    r11 = data["r11"]

    D4 = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]
    )

    D5 = np.array(
        [
            [1, 0, 0],
            [0, -1 / 2, -sqrt(3) / 2],
            [0, sqrt(3) / 2, -1 / 2],
        ]
    )

    v1 = np.array(
        [
            [r0, r1, -r1 / sqrt(3)],
            [r2, r11, r12],
            [-r1 / sqrt(3), r12, r11 + 2 * sqrt(3) / 3 * r12],
        ]
    )

    v4 = np.array(
        [
            [r0, r2, -r2 / sqrt(3)],
            [r1, r11, r12],
            [-r2 / sqrt(3), r12, r11 + 2 * sqrt(3) / 3 * r12],
        ]
    )

    v2 = D5 @ v4 @ D5.T
    v3 = D4 @ v1 @ D4.T
    v5 = D5 @ v1 @ D5.T
    v6 = D4 @ v4 @ D4.T

    return v1, v2, v3, v4, v5, v6


def IR(data):
    e1 = data["e1"]
    e2 = data["e2"]
    t0 = data["t0"]
    t1 = data["t1"]
    t2 = data["t2"]
    t12 = data["t12"]
    t11 = data["t11"]
    t22 = data["t22"]

    D_C3 = np.array(
        [
            [1, 0, 0],
            [0, cos(-2 * pi / 3), -sin(-2 * pi / 3)],
            [0, sin(-2 * pi / 3), cos(-2 * pi / 3)],
        ]
    )

    D_2C3 = np.array(
        [
            [1, 0, 0],
            [0, cos(-4 * pi / 3), -sin(-4 * pi / 3)],
            [0, sin(-4 * pi / 3), cos(-4 * pi / 3)],
        ]
    )

    D_S = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]
    )

    D_S1 = np.array(
        [
            [1, 0, 0],
            [0, 1 / 2, -sqrt(3) / 2],
            [0, -sqrt(3) / 2, -1 / 2],
        ]
    )

    D_S2 = np.array(
        [
            [1, 0, 0],
            [0, 1 / 2, sqrt(3) / 2],
            [0, sqrt(3) / 2, -1 / 2],
        ]
    )

    E_R0 = np.array(
        [
            [e1, 0, 0],
            [0, e2, 0],
            [0, 0, e2],
        ]
    )

    E_R1 = np.array(
        [
            [t0, t1, t2],
            [-t1, t11, t12],
            [t2, -t12, t22],
        ]
    )

    E_R2 = D_S1 @ E_R1 @ D_S1.T
    E_R3 = D_C3 @ E_R1 @ D_C3.T
    E_R4 = D_S @ E_R1 @ D_S.T
    E_R5 = D_2C3 @ E_R1 @ D_2C3.T
    E_R6 = D_S2 @ E_R1 @ D_S2.T

    return (E_R0, E_R1, E_R2, E_R3, E_R4, E_R5, E_R6)
