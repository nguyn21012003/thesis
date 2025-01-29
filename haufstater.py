import numpy as np
from numpy import cos, sin, pi, sqrt, exp

import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import subprocess

np.set_printoptions(precision=10, linewidth=1200, edgeitems=12)


alattice = 0.3190
e1 = 1.046
e2 = 2.104
t0 = -0.184
t1 = 0.401
t2 = 0.507
t11 = 0.218
t12 = 0.338
t22 = 0.057

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


def H(p, q, kx, ky):

    alpha = p / (2 * q)

    h0 = np.zeros([q, q], dtype=complex)

    for m in range(0, q):

        h0[m, m] = E_R0[0][0]

        h0[m, pbc(m + 1, q)] = E_R2[0][0] * exp(-1j * 2 * pi * (m + 1) * alpha) + E_R6[0][0] * exp(1j * 2 * pi * (m + 1) * alpha)

        h0[m, pbc(m - 1, q)] = E_R5[0][0] * exp(1j * 2 * pi * (m + 1) * alpha) + E_R3[0][0] * exp(-1j * 2 * pi * (m + 1) * alpha)

        h0[m, pbc(m + 2, q)] = E_R1[0][0]
        h0[m, pbc(m - 2, q)] = E_R4[0][0]

    return h0


def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


def pbc(i, q):
    if i == -1:
        return q - 1
    elif i == q:
        return 0
    elif i == q + 1:
        return 1
    elif i == q + 2:
        return 2
    else:
        return i


def plot_butterfly(q_max):
    file = f"1band_dataHofstadterButterfly_q_{q_max}.dat"

    with open(file, "w", newline="") as wfile:
        header = ["Alpha", "Energy", f"p/{q_max}"]
        writer = csv.DictWriter(wfile, fieldnames=header, delimiter=",")
        writer.writeheader()

        for p in tqdm(range(1, q_max + 1)):

            if gcd(p, q_max) == 1:

                alpha = p / q_max
                y = np.zeros(q_max)
                y[:] = alpha
                Ham = H(p, q_max, kx=0, ky=0)
                eigenvalue1 = np.linalg.eigvalsh(Ham)
                # print(Ham)

                for i in range(len(eigenvalue1)):
                    writer.writerow(
                        {
                            "Alpha": alpha,
                            "Energy": eigenvalue1[i],
                            f"p/{q_max}": f"{p}/{q_max}",
                        }
                    )

    filegnu = f"1band_plotHofstadterButterfly_q={q_max}.gnuplot"
    print("file data", file)
    print("file gnuplot: ", filegnu)
    with open(filegnu, "w") as gnuplotfile:
        gnuplotfile.write(
            f"""
set terminal wxt size 700,900 
set datafile separator ','
set title 'Hofstadter Butterfly'
set xlabel 'Alpha'
set ylabel 'Energy'
set grid
#set xrange [0:0.12]
#set yrange [1.3:2.0]
plot '{file}' u 1:2 with points pt 7 ps 0.3 lc rgb 'black' notitle 'HofstadterButterfly'
pause -1
        """
        )
    subprocess.run(["gnuplot", filegnu])

    # plt.plot(y, x1, "o", c="black", markersize=0.3)

    # plt.show()

    return


q_max = 151


plot_butterfly(q_max)
