import numpy as np
import csv
from numpy import linalg as LA
from numpy import real as RE
from matplotlib import pyplot as plt
from numpy import pi, cos, sin, sqrt
from tqdm import tqdm
import subprocess


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
        [
            a**3 * gamma4 * tau * kx * (kx**2 - 3 * ky**2),
            a**3 * gamma6 * (kx**2 + ky**2) * (tau * kx - 1j * ky),
        ],
        [
            a**3 * gamma6 * (kx**2 + ky**2) * (tau * kx + 1j * ky),
            a**3 * gamma5 * tau * kx * (kx**2 - 3 * ky**2),
        ],
    ]
    Hkp3 = sumMatrix(H2, H3)
    return Hkp3


def eigenvalue(fileValence, fileConduction):
    eigenvalues = []
    N = 100
    a_lattice = 3.190  ### Lattice constant cho hamiltonian kp nhieu loan bac 1,2,3
    Delta = 1.663  ### Delta cho hamiltonian kp nhieu loan bac 1,2,3
    tarr = [1.105, 1.059, 1.003]  ### t1, t2, t3 cho hamiltonian kp nhieu loan bac 1,2,3
    gammaArr2 = [
        0.055,
        0.077,
        -0.123,
    ]  ### Gamma1, Gamma2, Gamma3 cho hamiltonian kp nhieu loan bac 2
    gammaArr3 = [
        0.196,
        -0.065,
        -0.248,
        0.163,
        -0.094,
        -0.232,
    ]  ### Gamma1, Gamma2, Gamma3, Gamma4, Gamma5, Gamma6 cho hamiltonian kp nhieu loan bac 3

    tau = 1  ### Vi tri K hoac -K

    kx = np.linspace(-0.1 * 2 * pi / a_lattice, 0.1 * 2 * pi / a_lattice, N)
    # kx = np.linspace(-0.1, 0.1, N)
    # ky = np.linspace(0, 0, N)
    ky = 0

    V1 = np.zeros([N, N])
    C1 = np.zeros([N, N])
    V2 = np.zeros([N, N])
    C2 = np.zeros([N, N])
    V3 = np.zeros([N, N])
    C3 = np.zeros([N, N])

    for i in tqdm(range(N), desc="Proccesing", unit="step", ascii=" #"):
        for j in range(N):

            Hkp_1 = Hkp1(Delta, a_lattice, tarr[0], kx[i], ky, tau)
            Hkp_2 = Hkp2(
                Delta,
                a_lattice,
                tarr[1],
                kx[i],
                ky,
                gammaArr2[0],
                gammaArr2[1],
                gammaArr2[2],
                tau,
            )
            Hkp_3 = Hkp3(
                Delta,
                a_lattice,
                tarr[2],
                kx[i],
                ky,
                gammaArr3[0],
                gammaArr3[1],
                gammaArr3[2],
                gammaArr3[3],
                gammaArr3[4],
                gammaArr3[5],
                tau,
            )
            w1 = LA.eigvals(Hkp_1)
            w2 = LA.eigvals(Hkp_2)
            w3 = LA.eigvals(Hkp_3)

            C1[i, j] = RE(w1[0])  # - 0.831532225995396
            V1[i, j] = RE(w1[1])  # - -0.8315295742630259
            C2[i, j] = RE(w2[0])  # - 0.831532225995396
            V2[i, j] = RE(w2[1])  # - -0.8315295742630259
            C3[i, j] = RE(w3[0])  # - 0.831532225995396
            V3[i, j] = RE(w3[1])  # - -0.8315295742630259

    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(122)

    ax1.plot(kx, V1, color="blue", linestyle="--", label=r"$k^1$")
    ax1.plot(kx, V2, color="red", linestyle="-", label=r"$k^2$")
    ax1.plot(kx, V3, color="black", linestyle="-", label=r"$k^3$")
    ax1.plot(kx, C1, color="blue", linestyle="--", label=r"$k^1$")
    ax1.plot(kx, C2, color="red", linestyle="-", label=r"$k^2$")
    ax1.plot(kx, C3, color="black", linestyle="-", label=r"$k^3$")
    ax1.grid(True, axis="x")
    # ax1.set_ylim(-0.25, 0.001)
    ax1.set_xlim(-0.2, 0.2)
    ax1.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax1.set_xticklabels(["-0.1", r"($\Gamma$←)", r"K", r"(→M)", "0.1"])

    legend_without_duplicate_labels(ax1)

    # ax2.plot(kx, C1, color="blue", linestyle="--")
    # ax2.plot(kx, C2, color="red", linestyle="-")
    # ax2.plot(kx, C3, color="black", linestyle="-")
    # ax2.grid(True, axis="x")
    # ax2.set_ylim(-0.0005, 0.30)
    # ax2.set_xlim(-0.2, 0.2)
    # ax2.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    # ax2.set_xticklabels(["-0.1", r"($\Gamma$←)", r"K", r"(→M)", "0.1"])
    #
    with open(fileValence, "w") as writeValence:
        header = ["kx", "V1", "V2", "V3"]
        writer = csv.DictWriter(writeValence, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for i in range(N):
            for j in range(N):
                writer.writerow(
                    {
                        "kx": kx[i],
                        "V1": V1[i][j],
                        "V2": V2[i][j],
                        "V3": V3[i][j],
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
                        "C1": C1[i][j],
                        "C2": C2[i][j],
                        "C3": C3[i][j],
                    }
                )

    plt.show()


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
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
#set xrange [-0.1:0.1]

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
