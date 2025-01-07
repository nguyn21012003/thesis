import numpy as np
import csv
from numpy import linalg as LA
from numpy import real as RE
from matplotlib import pyplot as plt
from numpy import pi, cos, sin, sqrt
from tqdm import tqdm
import subprocess

a_lattice = 3.190  ### Lattice constant cho hamiltonian kp nhieu loan bac 1,2,3
Delta = 1.663  ### Delta cho hamiltonian kp nhieu loan bac 1,2,3
tarr = [1.105, 1.059, 1.003]  ### t1, t2, t3 cho hamiltonian kp nhieu loan bac 1,2,3
gammaArr2 = [0.055, 0.077, -0.123]  ### Gamma1, Gamma2, Gamma3 cho hamiltonian kp nhieu loan bac 2
gammaArr3 = [0.196, -0.065, -0.248, 0.163, -0.094, -0.232]  ### Gamma1, Gamma2, Gamma3, Gamma4, Gamma5, Gamma6 cho hamiltonian kp nhieu loan bac 3

tau = 1  ### Vi tri K hoac -K


hbar = 6.6e-1
B1 = 10
B2 = 0
eB1 = 10e-3 * B1
eB2 = 10e-3 * B2
me = 5.68


def sumMatrix(Matrix1, Matrix2):
    res = [[0 for _ in range(len(Matrix1[0]))] for _ in range(len(Matrix1))]

    for i in range(len(Matrix1)):
        for j in range(len(Matrix1[0])):
            res[i][j] = Matrix1[i][j] + Matrix2[i][j]
    return res


def Hkp1(Delta, a, t, kx, ky, tau):
    Hkp1 = [
        [Delta / 2, a * t * (tau * kx - 1j * ky)],
        [a * t * (tau * kx + 1j * ky), -Delta / 2],
    ]
    return Hkp1


def Hkp2(Delta, a, t, kx, ky, gamma1, gamma2, gamma3, tau):
    H1 = Hkp1(Delta, a, t, kx, ky, tau)
    H2 = [
        [a**2 * gamma1 * (kx**2 + ky**2), a**2 * gamma3 * (tau * kx + 1j * ky) ** 2],
        [a**2 * gamma3 * (tau * kx - 1j * ky) ** 2, a**2 * gamma2 * (kx**2 + ky**2)],
    ]
    Hkp2 = sumMatrix(H1, H2)
    return Hkp2


def Hkp3(Delta, a, t, kx, ky, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, tau):
    H2 = Hkp2(Delta, a, t, kx, ky, gamma1, gamma2, gamma3, tau)
    H3 = [
        [a**3 * gamma4 * tau * kx * (kx**2 - 3 * ky**2), a**3 * gamma6 * (kx**2 + ky**2) * (tau * kx - 1j * ky)],
        [a**3 * gamma6 * (kx**2 + ky**2) * (tau * kx + 1j * ky), a**3 * gamma5 * tau * kx * (kx**2 - 3 * ky**2)],
    ]
    Hkp3 = sumMatrix(H2, H3)
    return Hkp3


def eigenvalue(fileValence, fileConduction):
    eigenvalues = []
    N = 100
    eta = (eB1 / (8 * hbar)) * a_lattice**2 * sqrt(3)
    g_fact1 = eB1 * hbar * (1 + cos(eta)) / me

    kx = np.linspace(-0.1 * 2 * pi / a_lattice, 0.1 * 2 * pi / a_lattice, N)
    # kx = np.linspace(-0.1, 0.1, N)
    # ky = np.linspace(0, 0, N)
    ky = 0

    I2 = np.identity(2)
    pauliZ = np.array([[1, 0], [0, -1]])
    Hamiltonian_Zeeman = g_fact1 * np.kron(pauliZ, I2)

    V1_H1 = np.zeros([N, N])
    V2_H1 = np.zeros([N, N])
    C1_H1 = np.zeros([N, N])
    C2_H1 = np.zeros([N, N])

    V1_H2 = np.zeros([N, N])
    V2_H2 = np.zeros([N, N])
    C1_H2 = np.zeros([N, N])
    C2_H2 = np.zeros([N, N])

    V1_H3 = np.zeros([N, N])
    V2_H3 = np.zeros([N, N])
    C1_H3 = np.zeros([N, N])
    C2_H3 = np.zeros([N, N])

    for i in tqdm(range(N), desc="Proccesing", unit="step", ascii=" #"):
        for j in range(N):

            Hkp_1 = Hkp1(Delta, a_lattice, tarr[0], kx[i], ky, tau)
            Hkp_2 = Hkp2(Delta, a_lattice, tarr[1], kx[i], ky, gammaArr2[0], gammaArr2[1], gammaArr2[2], tau)
            Hkp_3 = Hkp3(
                Delta, a_lattice, tarr[2], kx[i], ky, gammaArr3[0], gammaArr3[1], gammaArr3[2], gammaArr3[3], gammaArr3[4], gammaArr3[5], tau
            )

            w1 = LA.eigvals(np.kron(Hkp_1, I2))
            w2 = LA.eigvals(np.kron(Hkp_2, I2))
            w3 = LA.eigvals(np.kron(Hkp_3, I2))

            C1_H1[i, j] = RE(w1[0])
            C2_H1[i, j] = RE(w1[2])
            V1_H1[i, j] = RE(w1[1])
            V2_H1[i, j] = RE(w1[3])

            C1_H2[i, j] = RE(w2[0])
            C2_H2[i, j] = RE(w2[2])
            V1_H2[i, j] = RE(w2[1])
            V2_H2[i, j] = RE(w2[3])

            C1_H3[i, j] = RE(w3[0])
            C2_H3[i, j] = RE(w3[2])
            V1_H3[i, j] = RE(w3[1])
            V2_H3[i, j] = RE(w3[3])

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    hex_code = "#000000"
    hex_code1 = None
    # plt.subplot(1, 3, 3)
    ax1.plot(kx, V1_H1, color="blue", linestyle="--", label=r"$k^1$")
    ax1.plot(kx, V2_H1, color="blue", linestyle="--", label=r"$k^1$")
    ax1.plot(kx, V1_H2, color="red", linestyle="-", label=r"$k^2$")
    ax1.plot(kx, V2_H2, color="red", linestyle="-", label=r"$k^2$")
    ax1.plot(kx, V1_H3, color="black", linestyle="-", label=r"$k^3$")
    ax1.plot(kx, V2_H3, color="black", linestyle="-", label=r"$k^3$")
    ax1.grid(True, axis="x")
    legend_without_duplicate_labels(ax1)

    ax2.plot(kx, C1_H1, color="blue", linestyle="--", label=r"$k^1$")
    ax2.plot(kx, C1_H2, color="red", linestyle="-", label=r"$k^2$")
    ax2.plot(kx, C1_H3, color="black", linestyle="-", label=r"$k^3$")
    ax2.grid(True, axis="x")
    legend_without_duplicate_labels(ax2)

    with open(fileValence, "w") as writeValence:
        header = ["kx", "V1", "V2", "V3"]
        writer = csv.DictWriter(writeValence, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for i in range(N):
            for j in range(N):
                writer.writerow(
                    {
                        "kx": kx[i],
                        "V1": V1_H1[i][j],
                        "V2": V1_H2[i][j],
                        "V3": V1_H3[i][j],
                    }
                )
    with open(fileConduction, "w") as writeConduction:
        header = ["kx", "C1", "C2", "C3"]
        writer = csv.DictWriter(writeConduction, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for i in range(N):
            for j in range(N):
                writer.writerow(
                    {
                        "kx": kx[i],
                        "C1": C1_H1[i][j],
                        "C2": C1_H2[i][j],
                        "C3": C1_H3[i][j],
                    }
                )

    # plt.gca().set_xticks([-4 * pi / (3 * a_lattice), 0, 4 * pi / (3 * a_lattice)], minor=True)
    plt.xlabel("kx")
    plt.ylabel("eV")
    # plt.xticks([-2 * pi / a_lattice, -4 * pi / (3 * a_lattice), 0, 4 * pi / (3 * a_lattice), 2 * pi / a_lattice], ["M", "K", "Γ", "K", "M"])

    plt.show()


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


def gnuPlot(fileV, fileC):
    """
    Không xài dòng nào thì # vào dòng đó
    """

    with open("gnuPlot.gp", "w") as gnuplot:
        gnuplot.write(
            f"""
            
set multiplot layout 1, 2
set grid


set style line 1 linetype 1 dashtype 2 lc rgb "blue" lw 3
set style line 2 linetype 1 lc rgb "red" lw 3
set style line 3 linetype 1 lc rgb "black" lw 3
set xrange [-0.1:0.1]

plot "{fileV}" using 1:2 with lines linestyle 1 title "k^1", \
     "{fileV}" using 1:3 with lines linestyle 2 title "k^2", \
     "{fileV}" using 1:4 with lines linestyle 3 title "k^3"

plot "{fileC}" using 1:2 with lines linestyle 1 title "k^1", \
     "{fileC}" using 1:3 with lines linestyle 2 title "k^2", \
     "{fileC}" using 1:4 with lines linestyle 3 title "k^3", 

pause -1
"""
        )
    subprocess.run(["gnuplot", "gnuPlot.gp"])


def main():
    fileValence = "eigenvaluesValence.csv"
    fileConduction = "eigenvaluesConduction.csv"
    eigenvalue(fileValence, fileConduction)
    # gnuPlot(fileValence, fileConduction)


if __name__ == "__main__":
    main()
