import numpy as np
import csv
from numpy import pi, exp
from numpy import linalg as LA
from math import sqrt, cos, sin
import matplotlib as mpl
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm


def para(argument):
    argument = int(argument)
    matt = ["MoS2", "WS2", "MoSe2", "WSe2", "MoTe2", "WTe2"]
    a = [3.190, 3.191, 3.326, 3.325, 3.357, 3.560]
    e1 = [1.046, 1.130, 0.919, 0.943, 0.605, 0.606]
    e2 = [2.104, 2.275, 2.065, 2.179, 1.972, 2.102]
    t0 = [-0.184, -0.206, -0.188, -0.207, -0.169, -0.175]
    t1 = [0.401, 0.567, 0.317, 0.457, 0.228, 0.342]
    t2 = [0.507, 0.536, 0.456, 0.486, 0.390, 0.410]
    t11 = [0.218, 0.286, 0.211, 0.263, 0.207, 0.233]
    t12 = [0.338, 0.384, 0.290, 0.329, 0.239, 0.270]
    t22 = [0.057, -0.061, 0.130, 0.034, 0.252, 0.190]
    match argument:
        case argument:
            return (
                matt[argument],
                a[argument],
                e1[argument],
                e2[argument],
                t0[argument],
                t1[argument],
                t2[argument],
                t11[argument],
                t12[argument],
                t22[argument],
            )


# IR

D_C3 = np.array(
    [
        [1, 0, 0],
        [0, -1 / 2, sqrt(3) / 2],
        [0, -sqrt(3) / 2, -1 / 2],
    ]
)

D_2C3 = np.array(
    [
        [1, 0, 0],
        [0, -1 / 2, -sqrt(3) / 2],
        [0, sqrt(3) / 2, -1 / 2],
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


def Hamiltonian(argument, p, q, kx, ky):
    matt, alattice, e1, e2, t0, t1, t2, t11, t12, t22 = para(argument)
    alpha = p / q
    a = 1 / 2 * kx * alattice
    b = sqrt(3) / 2 * ky * alattice
    h0 = np.zeros([q, q], dtype=complex)
    h1 = np.zeros([q, q], dtype=complex)
    h1T = np.zeros([q, q], dtype=complex)
    h2 = np.zeros([q, q], dtype=complex)
    h2T = np.zeros([q, q], dtype=complex)
    h11 = np.zeros([q, q], dtype=complex)
    h22 = np.zeros([q, q], dtype=complex)
    h12 = np.zeros([q, q], dtype=complex)
    h12T = np.zeros([q, q], dtype=complex)

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

    for m in range(0, q):

        h0[m][m] = E_R0[0][0]

        h1[m][m] = (
            E_R0[0][1]
            + E_R2[0][1] * exp(-1j * 2 * pi * m * alpha) * exp(1j * (a - b))
            + E_R3[0][1] * exp(-1j * 2 * pi * m * alpha) * exp(1j * (-a - b))
            + E_R5[0][1] * exp(1j * 2 * pi * m * alpha) * exp(1j * (-a + b))
            + E_R6[0][1] * exp(1j * 2 * pi * m * alpha) * exp(1j * (a + b))
        )

        h1T[m][m] = (
            E_R0[1][0]
            + E_R2[1][0] * exp(1j * 2 * pi * m * alpha) * exp(-1j * (a - b))
            + E_R3[1][0] * exp(1j * 2 * pi * m * alpha) * exp(-1j * (-a - b))
            + E_R5[1][0] * exp(-1j * 2 * pi * m * alpha) * exp(-1j * (-a + b))
            + E_R6[1][0] * exp(-1j * 2 * pi * m * alpha) * exp(-1j * (a + b))
        )

        h2[m][m] = (
            E_R0[0][2]
            + E_R2[0][2] * exp(-1j * 2 * pi * m * alpha) * exp(1j * (a - b))
            + E_R3[0][2] * exp(-1j * 2 * pi * m * alpha) * exp(1j * (-a - b))
            + E_R5[0][2] * exp(1j * 2 * pi * m * alpha) * exp(1j * (-a + b))
            + E_R6[0][2] * exp(1j * 2 * pi * m * alpha) * exp(1j * (a + b))
        )

        h2T[m][m] = (
            E_R0[2][0]
            + E_R2[2][0] * exp(1j * 2 * pi * m * alpha) * exp(-1j * (a - b))
            + E_R3[2][0] * exp(1j * 2 * pi * m * alpha) * exp(-1j * (-a - b))
            + E_R5[2][0] * exp(-1j * 2 * pi * m * alpha) * exp(-1j * (-a + b))
            + E_R6[2][0] * exp(-1j * 2 * pi * m * alpha) * exp(-1j * (a + b))
        )

        h11[m][m] = (
            E_R0[1][1]
            + E_R2[1][1] * exp(-1j * 2 * pi * m * alpha) * exp(1j * (a - b))
            + E_R3[1][1] * exp(-1j * 2 * pi * m * alpha) * exp(1j * (-a - b))
            + E_R5[1][1] * exp(1j * 2 * pi * m * alpha) * exp(1j * (-a + b))
            + E_R6[1][1] * exp(1j * 2 * pi * m * alpha) * exp(1j * (a + b))
        )

        h12[m][m] = (
            E_R0[1][2]
            + E_R2[1][2] * exp(-1j * 2 * pi * m * alpha) * exp(1j * (a - b))
            + E_R3[1][2] * exp(-1j * 2 * pi * m * alpha) * exp(1j * (-a - b))
            + E_R5[1][2] * exp(1j * 2 * pi * m * alpha) * exp(1j * (-a + b))
            + E_R6[1][2] * exp(1j * 2 * pi * m * alpha) * exp(1j * (a + b))
        )

        h12T[m][m] = (
            E_R0[2][1]
            + E_R2[2][1] * exp(1j * 2 * pi * m * alpha) * exp(-1j * (a - b))
            + E_R3[2][1] * exp(1j * 2 * pi * m * alpha) * exp(-1j * (-a - b))
            + E_R5[2][1] * exp(-1j * 2 * pi * m * alpha) * exp(-1j * (-a + b))
            + E_R6[2][1] * exp(-1j * 2 * pi * m * alpha) * exp(-1j * (a + b))
        )

        h22[m][m] = (
            E_R0[2][2]
            + E_R2[2][2] * exp(-1j * 2 * pi * m * alpha) * exp(1j * (a - b))
            + E_R3[2][2] * exp(-1j * 2 * pi * m * alpha) * exp(1j * (-a - b))
            + E_R5[2][2] * exp(1j * 2 * pi * m * alpha) * exp(1j * (-a + b))
            + E_R6[2][2] * exp(1j * 2 * pi * m * alpha) * exp(1j * (a + b))
        )

        if m == 0:  #### First row

            h0[m][m + 1] = E_R2[0][0] * exp(2 * 1j * pi * m * alpha - 1j * ky * alattice) + E_R6[0][0] * exp(
                -2 * 1j * pi * m * alpha + 1j * ky * alattice
            )

        elif m == q - 1:  ### last Row

            h1[m][m - 1] = E_R4[0][1]

            h1T[m][m - 1] = E_R4[1][0]

            h2[m][m - 1] = E_R4[0][2]

            h2T[m][m - 1] = E_R4[2][0]

            h11[m][m - 1] = E_R4[1][1]

            h12[m][m - 1] = E_R4[1][2]

            h12T[m][m - 1] = E_R4[2][1]

            h22[m][m - 1] = E_R4[2][2]

        else:

            h0[m][m + 1] = E_R1[0][0]  ## thành phần bên trái đường chéo
            h0[m][m - 1] = E_R4[0][0]  ## thành phần bên phải đường chéo

            h1[m][m + 1] = E_R1[0][1]
            h1[m][m - 1] = E_R4[0][1]

            h1T[m][m + 1] = E_R1[1][0]
            h1T[m][m - 1] = E_R4[1][0]

            h2[m][m + 1] = E_R1[0][2]
            h2[m][m - 1] = E_R4[0][2]

            h2T[m][m + 1] = E_R1[2][0]
            h2T[m][m - 1] = E_R4[2][0]

            h11[m][m + 1] = E_R1[1][1]
            h11[m][m - 1] = E_R4[1][1]

            h22[m][m + 1] = E_R1[2][2]
            h22[m][m - 1] = E_R4[2][2]

            h12[m][m + 1] = E_R1[1][2]
            h12[m][m - 1] = E_R4[1][2]

            h12T[m][m + 1] = E_R1[2][1]
            h12T[m][m - 1] = E_R4[2][1]

    H = np.zeros([3 * q, 3 * q], dtype=complex)
    # H = np.zeros([q, q], dtype=complex)
    # H = np.zeros([2 * q, 2 * q], dtype=complex)

    H[0:q, 0:q] = h0
    H[0:q, q : 2 * q] = h1
    H[0:q, 2 * q : 3 * q] = h2
    H[q : 2 * q, 0:q] = h1T
    H[q : 2 * q, q : 2 * q] = h11
    H[q : 2 * q, 2 * q : 3 * q] = h12
    H[2 * q : 3 * q, 0:q] = h2T
    H[2 * q : 3 * q, q : 2 * q] = h12T
    H[2 * q : 3 * q, 2 * q : 3 * q] = h22

    ############ 1 band
    # H[0:q, 0:q] = h22

    ############ 3 band
    # H[0:q, 0:q] = h0
    # H[0:q, q : 2 * q] = h0
    # H[0:q, 2 * q : 3 * q] = h0
    # H[q : 2 * q, 0:q] = np.conjugate(h0).T
    # H[q : 2 * q, q : 2 * q] = h0
    # H[q : 2 * q, 2 * q : 3 * q] = h0
    # H[2 * q : 3 * q, 0:q] = np.conjugate(h0).T
    # H[2 * q : 3 * q, q : 2 * q] = np.conjugate(h0).T
    # H[2 * q : 3 * q, 2 * q : 3 * q] = h0

    ############### 2 band
    # H[0:q, 0:q] = h0
    # H[0:q, q : 2 * q] = h1
    # H[q : 2 * q, 0:q] = np.conjugate(h1).T
    # H[q : 2 * q, q : 2 * q] = h11

    return H


# Periodic boundary conditions
def pbc(i, q):
    if i == q:
        return 0
    elif i == -1:
        return q - 1
    else:
        return i


def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


def main():
    choice = 0  # int(input(("Input material: ")))
    qmax = 151 * 1  # int(input("Input the range q max aka the magnetic cell: "))
    matt, alattice, e1, e2, t0, t1, t2, t11, t12, t22 = para(choice)
    plt.figure(figsize=(7, 7))
    plt.title(f"{matt}")
    file = f"3band_dataHofstadterButterfly_q_{qmax}_{matt}_test.dat"
    # file = f"1band_dataHofstadterButterfly_q_{qmax}_h0.dat"
    # file = f"1band_dataHofstadterButterfly_q_{qmax}_h11.dat"
    # file = f"1band_dataHofstadterButterfly_q_{qmax}_h22.dat"
    kpoints = {
        "Γ": [0, 0],
        "K": [4 * pi / (3 * alattice), 0],
        "M": [pi / alattice, pi / (sqrt(3) * alattice)],
    }

    print(file)
    with open(file, "w", newline="") as writefile:
        header = ["Alpha", "Energy"]
        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        for p in tqdm(range(1, qmax + 1)):
            if gcd(p, qmax) == 1:
                alpha = p / qmax
                y = np.zeros(3 * qmax)
                y[:] = alpha

                eigenvalue1 = LA.eigvalsh(Hamiltonian(choice, p, qmax, kx=kpoints["K"][0], ky=kpoints["K"][1]))
                for i in range(len(eigenvalue1)):
                    writer.writerow(
                        {
                            "Alpha": alpha,
                            "Energy": eigenvalue1[i],
                        }
                    )

                plt.plot(y, eigenvalue1, "o", c="red", markersize=0.1)

    filegnu = f"plotHofstadterButterfly_q={qmax}_{matt}.gnuplot"
    with open(filegnu, "w") as gnuplotfile:
        gnuplotfile.write(
            f"""
set datafile separator ','
set title 'Hofstadter Butterfly'
set xlabel 'Alpha'
set ylabel 'Energy'
set grid
#set xrange [0:0.12]
#set yrange [1.3:2.0]
plot '{file}' u 1:2 with points pt 7 ps 0.3 lc rgb 'black' notitle
pause -1
        """
        )
    subprocess.run(["gnuplot", filegnu])

    # plt.plot(y, eigenvalue2, "o", c="red", markersize=0.1)

    plt.show()
    plt.tight_layout()
    # plt.savefig(f"{matt}_qmax_{qmax}.png")


if __name__ == "__main__":
    main()
