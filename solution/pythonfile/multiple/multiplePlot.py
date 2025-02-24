import csv
import subprocess
from math import cos, sin, sqrt

import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from numpy import linalg as LA
from numpy import pi
from tqdm import tqdm

np.set_printoptions(precision=10, linewidth=1200, edgeitems=20, suppress=True)


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
def IR(e1, e2, t0, t1, t2, t11, t12, t22):
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

    return E_R0, E_R1, E_R2, E_R3, E_R4, E_R5, E_R6


def Hamiltonian(argument, alattice, p, q, kx, ky, E_R0, E_R1, E_R2, E_R3, E_R4, E_R5, E_R6):
    # matt, alattice, e1, e2, t0, t1, t2, t11, t12, t22 = para(argument)
    eta = p / q
    alpha = 1 / 2 * kx * alattice
    beta = sqrt(3) / 2 * ky * alattice
    h0 = np.zeros([q, q], dtype=complex)
    h1 = np.zeros([q, q], dtype=complex)
    h1T = np.zeros([q, q], dtype=complex)
    h2 = np.zeros([q, q], dtype=complex)
    h2T = np.zeros([q, q], dtype=complex)
    h11 = np.zeros([q, q], dtype=complex)
    h22 = np.zeros([q, q], dtype=complex)
    h12 = np.zeros([q, q], dtype=complex)
    h12T = np.zeros([q, q], dtype=complex)
    H = np.zeros([3 * q, 3 * q], dtype=complex)

    for m in range(0, q):
        h0[m][m] = E_R0[0][0]
        h1[m][m] = E_R0[0][1]
        h2[m][m] = E_R0[0][2]
        h1T[m][m] = E_R0[1][0]
        h11[m][m] = E_R0[1][1]
        h12[m][m] = E_R0[1][2]
        h2T[m][m] = E_R0[2][0]
        h12T[m][m] = E_R0[2][1]
        h22[m][m] = E_R0[2][2]

        h0[m, pbc(m + 2, q)] = E_R1[0][0]
        h1[m, pbc(m + 2, q)] = E_R1[0][1]
        h2[m, pbc(m + 2, q)] = E_R1[0][2]
        h1T[m, pbc(m + 2, q)] = E_R1[1][0]
        h11[m, pbc(m + 2, q)] = E_R1[1][1]
        h12[m, pbc(m + 2, q)] = E_R1[1][2]
        h2T[m, pbc(m + 2, q)] = E_R1[2][0]
        h12T[m, pbc(m + 2, q)] = E_R1[2][1]
        h22[m, pbc(m + 2, q)] = E_R1[2][2]

        h0[m, pbc(m + 1, q)] = E_R2[0][0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * exp(-1j * beta) + E_R6[0][0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * exp(1j * beta)
        h1[m, pbc(m + 1, q)] = E_R2[0][1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * exp(-1j * beta) + E_R6[0][1] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * exp(1j * beta)
        h2[m, pbc(m + 1, q)] = E_R2[0][2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * exp(-1j * beta) + E_R6[0][2] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * exp(1j * beta)
        h1T[m, pbc(m + 1, q)] = E_R2[1][0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * exp(1j * beta) + E_R6[1][0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * exp(-1j * beta)
        h11[m, pbc(m + 1, q)] = E_R2[1][1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * exp(-1j * beta) + E_R6[1][1] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * exp(1j * beta)
        h12[m, pbc(m + 1, q)] = E_R2[1][2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * exp(-1j * beta) + E_R6[1][2] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * exp(1j * beta)
        h2T[m, pbc(m + 1, q)] = E_R2[2][0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * exp(1j * beta) + E_R6[2][0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * exp(-1j * beta)
        h12T[m, pbc(m + 1, q)] = E_R2[2][1] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * exp(1j * beta) + E_R6[2][1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * exp(-1j * beta)
        h22[m, pbc(m + 1, q)] = E_R2[2][2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * exp(-1j * beta) + E_R6[2][2] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * exp(1j * beta)

        h0[m, pbc(m - 1, q)] = E_R5[0][0] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * exp(1j * beta) + E_R3[0][0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * exp(-1j * beta)
        h1[m, pbc(m - 1, q)] = E_R5[0][1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * exp(1j * beta) + E_R3[0][1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * exp(-1j * beta)
        h2[m, pbc(m - 1, q)] = E_R5[0][2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * exp(1j * beta) + E_R3[0][2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * exp(-1j * beta)
        h1T[m, pbc(m - 1, q)] = E_R5[1][0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * exp(-1j * beta) + E_R3[1][0] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * exp(1j * beta)
        h11[m, pbc(m - 1, q)] = E_R5[1][1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * exp(1j * beta) + E_R3[1][1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * exp(-1j * beta)
        h12[m, pbc(m - 1, q)] = E_R5[1][2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * exp(1j * beta) + E_R3[1][2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * exp(-1j * beta)
        h2T[m, pbc(m - 1, q)] = E_R5[2][0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * exp(-1j * beta) + E_R3[2][0] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * exp(1j * beta)
        h12T[m, pbc(m - 1, q)] = E_R5[2][1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * exp(-1j * beta) + E_R3[2][1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * exp(1j * beta)
        h22[m, pbc(m - 1, q)] = E_R5[2][2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * exp(1j * beta) + E_R3[2][2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * exp(-1j * beta)

        h0[m, pbc(m - 2, q)] = E_R4[0][0]
        h1[m, pbc(m - 2, q)] = E_R4[0][1]
        h2[m, pbc(m - 2, q)] = E_R4[0][2]
        h1T[m, pbc(m - 2, q)] = E_R4[1][0]
        h11[m, pbc(m - 2, q)] = E_R4[1][1]
        h12[m, pbc(m - 2, q)] = E_R4[1][2]
        h2T[m, pbc(m - 2, q)] = E_R4[2][0]
        h12T[m, pbc(m - 2, q)] = E_R4[2][1]
        h22[m, pbc(m - 2, q)] = E_R4[2][2]

    H[0:q, 0:q] = h0
    H[0:q, q : 2 * q] = h1
    H[0:q, 2 * q : 3 * q] = h2
    H[q : 2 * q, 0:q] = h1T
    H[q : 2 * q, q : 2 * q] = h11
    H[q : 2 * q, 2 * q : 3 * q] = h12
    H[2 * q : 3 * q, 0:q] = h2T
    H[2 * q : 3 * q, q : 2 * q] = h12T
    H[2 * q : 3 * q, 2 * q : 3 * q] = h22

    return H


def PeierlsSubstitution(kx, ky, alattice, p, q):
    intergral = 2 * pi

    return intergral


def plotMatrix(matrix):
    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(matrix), cmap="binary", interpolation="none")
    plt.colorbar(label="magnitude")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()

    return None


def saveMatrix(matrix, fileName):

    with open(fileName, "w", newline="") as Matrixfile:
        header = ["row", "column", "value"]
        writer = csv.DictWriter(Matrixfile, fieldnames=header, delimiter=",")
        writer.writeheader()
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                Matrixfile.write(f"{i+1} {j+1} {np.abs(matrix[i,j])}\n")
            Matrixfile.write("\n")

    return None


def PlotMatrixGNU(fileMatrix, fileName):
    with open(fileName, "w") as GNUPLOT:
        GNUPLOT.write(
            f"""
set pm3d map
set size ratio -1
set palette defined (0 "white", 1 "blue", 2 "yellow", 3 "red") 
set xlabel "Column Index"
set ylabel "Row Index"
set yrange [*:*] reverse
set key top right font "Arial,10"
set tics out

set xtics offset 0,1

set autoscale xfix
set autoscale yfix


#set tics scale 0,0.001
#set mxtics 2
#set mytics 2
#set grid front mxtics mytics lw 1.5 lt -1 lc rgb 'white'



unset key
splot "{fileMatrix}" using 1:2:3 with image
pause -1
        """
        )
    # subprocess.run(["gnuplot", fileName])
    return None


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
    # return i % q


def process(choice):
    qmax = 797  # int(input("Input the range q max aka the magnetic cell: "))
    matt, alattice, e1, e2, t0, t1, t2, t11, t12, t22 = para(choice)
    E_R0, E_R1, E_R2, E_R3, E_R4, E_R5, E_R6 = IR(e1, e2, t0, t1, t2, t11, t12, t22)

    # fig, ax = plt.subplots(constrained_layout=True)
    # ax.margins(0)
    # plt.title(f"{matt}")

    file = f"3band_dataHofstadterButterfly_q_{qmax}_{matt}_final.dat"  ## Cần file này
    # file = f"2band_dataHofstadterButterfly_q_{qmax}_{matt}_final.dat"
    # file = f"1band_dataHofstadterButterfly_q_{qmax}_h0.dat"
    # file = f"1band_dataHofstadterButterfly_q_{qmax}_h11.dat"
    # file = f"1band_dataHofstadterButterfly_q_{qmax}_h22.dat"

    # fileMatrix = f"1band_Matrix_q_{qmax}_h0.dat"
    fileMatrix = f"3band_Matrix_q_{qmax}.dat"
    # file_plot_Matrix_Gnu = f"1band_Matrix_q_{qmax}_h0.gnuplot"
    file_plot_Matrix_Gnu = f"3band_Matrix_q_{qmax}_h0.gnuplot"

    # fileLandauLevels = f"LandauLevels_q={qmax}_{matt}_1_band.dat"
    fileLandauLevels = f"LandauLevels_q={qmax}_{matt}_3_band.dat"

    filegnu = f"plotHofstadterButterfly_q={qmax}_{matt}.gnuplot"  ## Cần File này
    print("\n")
    print("file data: ", file)
    print("file gnuplot: ", filegnu)
    print("file landau: ", fileLandauLevels)
    print("file Matrix: ", fileMatrix)
    print("file Matrix GNU: ", file_plot_Matrix_Gnu)

    kpoints = {
        "G": [0, 0],
        "K": [4 * pi / (3 * alattice), 0],
        "M": [pi / (alattice), pi / (sqrt(3) * alattice)],
    }

    print(file)
    kx = kpoints["K"][0]
    ky = kpoints["K"][1]
    with open(file, "w", newline="") as writefile:
        header = [
            "eta",
            "Energy",
            f"p/{qmax}",
        ]

        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        list_eigen = []
        for p in tqdm(range(1, qmax + 1), desc=f"{matt}", colour="green"):
            if gcd(p, qmax) == 1:
                eta = p / qmax
                # y = np.zeros(1 * qmax)
                # y[:] = eta
                H = Hamiltonian(choice, alattice, p, qmax, kx, ky, E_R0, E_R1, E_R2, E_R3, E_R4, E_R5, E_R6)

                eigenvalue1 = LA.eigvalsh(H)
                list_eigen.append(eigenvalue1)
                for i in range(len(eigenvalue1)):
                    writer.writerow(
                        {
                            "eta": eta,
                            "Energy": eigenvalue1[i],
                            f"p/{qmax}": f"{p}/{qmax}",
                        }
                    )

        saveMatrix(H, fileMatrix)
        PlotMatrixGNU(fileMatrix, file_plot_Matrix_Gnu)
        plotMatrix(H)

        # plt.plot(y, eigenvalue1, "o", c="black", markersize=0.1)

    n_levels = 11
    with open(fileLandauLevels, "w", newline="") as f:
        for p in range(1, qmax + 1):
            if gcd(p, qmax) == 1:
                alpha = p / qmax
                values = [alpha]
                for n in range(n_levels):
                    E = t0 * (6 - 8 * pi * sqrt(3) * alpha * (n + 0.5)) + e1
                    # E = -t0 * sqrt(2 * pi * sqrt(3) * n * alpha) + e1
                    values.append(E)

                f.write(",".join(map(str, values)) + "\n")

    with open(filegnu, "w") as gnuplotfile:
        gnuplotfile.write(
            f"""
set terminal wxt size 700,900
set datafile separator ','
#set title 'Hofstadter Butterfly'

set key top right font "Arial,10"
set key font ",25"
set xtics font ",13"
set ytics font ",13"
set xlabel 'p/q' font 'Arial,16'
set ylabel 'Energy' font 'Arial,16'
set grid

#set xrange [0:0.2]
#set yrange [-0.2:1.75]

set tics out
plot '{file}' u 1:2 with points pt 7 ps 0.03 lc rgb 'black' notitle 'HofstadterButterfly_{matt}' ,\
     #for [i=2:{n_levels}] "{fileLandauLevels}" using 1:i with lines lw 0.75 lc rgb 'black' notitle
pause -1
        """
        )
    # subprocess.run(["gnuplot", filegnu])

    # ax.tick_params(axis="both", direction="out")
    # plt.show()
    # plt.savefig(f"{matt}_qmax_{qmax}.png")

    return None


def main():
    itter = 0
    for i in tqdm(range(6), desc=f"Progress {itter}", colour="green"):

        process(i)
        itter += 1


if __name__ == "__main__":
    main()
