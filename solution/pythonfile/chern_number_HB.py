import csv
import subprocess
from math import cos, sin, sqrt

import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from numpy import linalg as LA
from numpy import pi
from numpy import mod
from tqdm import tqdm
from copy import deepcopy
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

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

    return h0


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
                Matrixfile.write(f"{i + 1} {j + 1} {np.abs(matrix[i, j])}\n")
            Matrixfile.write("\n")

    return None


def PlotMatrixGNU(fileMatrix, fileName):
    with open(fileName, "w") as GNUPLOT:
        GNUPLOT.write(
            f"""
set size ratio -1
set palette defined (0 "white", 1 "blue", 2 "yellow", 3 "red")
set xlabel "Column Index"
set ylabel "Row Index"
set yrange [*:*] reverse

set tics out

set xtics offset 0,1

set autoscale xfix
set autoscale yfix
set linestyle 81 lt 1 lw 9.0 lc rgb "black"

set xtics 20
set ytics 20


set grid


#set tics scale 0,0.001
#set mxtics 2
#set mytics 2
#set grid front mxtics mytics lw 1.5 lt -1 lc rgb 'white'

set label "(b)" at 2,55,10 font "Arial,16" front


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


def LandauLevels(
    fileLandauLevels_conductionBand,
    fileLandauLevels_valenceBand,
    qmax,
    n_levels,
    choice,
):
    matt, alattice, e1, e2, t0, t1, t2, t11, t12, t22 = para(choice)
    with open(fileLandauLevels_conductionBand, "w", newline="") as fc:
        for p1 in tqdm(range(1, qmax + 1), ascii=" #", desc="Landau Levels conduction"):
            if gcd(p1, qmax) == 1:
                alpha1 = p1 / qmax
                values_c = [alpha1]
                for n1 in range(n_levels):
                    E_con = t0 * (6 + 8 * pi * sqrt(3) * alpha1 * (n1 + 0.5)) + e1
                    # E = -t0 * sqrt(2 * pi * sqrt(3) * n * alpha) + e1
                    values_c.append(E_con)

                fc.write(",".join(map(str, values_c)) + "\n")

    with open(fileLandauLevels_valenceBand, "w", newline="") as fv:
        for p2 in tqdm(range(1, qmax + 1), ascii=" #", desc="Landau Levels valence"):
            if gcd(p2, qmax) == 1:
                alpha2 = p2 / qmax
                values_v = [alpha2]
                for n2 in range(n_levels):
                    E_val = 3 * (t11 + t22) * (n2 + 1 / 2) ** (12) + e2

                    values_v.append(E_val)
                fv.write(",".join(map(str, values_v)) + "\n")


def chern(p, q):
    sr_list, tr_list, kj_list = [], [], []
    chern_number_data_list = {
        "sr_list": [],
        "tr_list": [],
        "kj_list": [],
    }
    for r in range(q + 1):
        if q % 2 == 0 and r == q / 2:
            continue
        for tr in range(-int(q / 2), int(q / 2) + 1):
            for sr in range(-q, q + 1):
                if r == q * sr + p * tr:
                    sr_list.append(sr)
                    tr_list.append(tr)
                    kj_list.append(q // 2)
                    break
            else:
                continue
            break

    Chern_list = []
    if q % 2 != 0:
        numb_band_groups = q
    else:
        numb_band_groups = q - 1

    for i in range(numb_band_groups):
        Chern_list.append(tr_list[i + 1] - tr_list[i])

    if q % 2 == 0:
        Chern_list.insert(q // 2 - 1, Chern_list[q // 2 - 1])

    return Chern_list, tr_list


def process(choice, qmax):
    matt, alattice, e1, e2, t0, t1, t2, t11, t12, t22 = para(choice)
    E_R0, E_R1, E_R2, E_R3, E_R4, E_R5, E_R6 = IR(e1, e2, t0, t1, t2, t11, t12, t22)

    kpoints = {
        "G": [0, 0],
        "K": [4 * pi / (3 * alattice), 0],
        "M": [pi / (alattice), pi / (sqrt(3) * alattice)],
    }
    data = {"E_list": [], "eta_list": [], "chern_list": [], "tr_list": [], "kj_list": [], "E_list_orig": [], "matrix": None, "matt": [], "gaps_list": [], "p_list": [], "tr_DOS": [], "eta_DOS": [], "DOS_list": []}
    data["matt"] = matt
    kx = kpoints["K"][0]
    ky = kpoints["K"][1]
    for p in tqdm(range(1, qmax + 1), ascii=" #", desc=f"{matt}"):
        if gcd(p, qmax) != 1:
            continue
        eta = p / qmax
        H = Hamiltonian(choice, alattice, p, qmax, kx, ky, E_R0, E_R1, E_R2, E_R3, E_R4, E_R5, E_R6)
        const = len(H) // qmax
        data["eta_list"].append([eta] * const * qmax)
        clist, trlist = chern(p, const * qmax)
        data["chern_list"].append(clist)
        data["tr_list"].append(trlist)
        eigenvalue = np.sort(LA.eigvalsh(H))
        data["E_list"].append(eigenvalue)

        resx = np.shape(data["E_list"])[0]
        resy = np.shape(data["E_list"])[1]
        res = [resx, resy]
        E_vals = np.linspace(-0.0025443531460326053, 2.5, res[1])
        data["matrix"] = np.zeros((res[0], res[1]))
        data["p_list"].append(p)

        data["eta_DOS"].append([eta] * (len(H) - 1))
        data["DOS_list"].append([i / len(H) for i in range(len(H) - 1)])
        data["gaps_list"].append([eigenvalue[i + 1] - eigenvalue[i] for i in range(len(H) - 1)])
        data["tr_DOS"].append(trlist[1:-1])

        data["E_list_orig"] = deepcopy(data["E_list"])
        for i, p in enumerate(range(1, res[0] + 1)):
            for j, E in enumerate(E_vals):
                for k, El in enumerate(data["E_list"][i]):
                    if E <= El:
                        data["matrix"][i][j] = data["tr_list"][i][k]
                        break

    print(E_vals, "\n")
    print(data["matrix"], "\n")
    return data


def plotPython(data):
    E_list = data["E_list"]
    E_list_orig = data["E_list_orig"]
    eta_list = data["eta_list"]
    clist = data["chern_list"]
    tlist = data["tr_list"]
    klist = data["kj_list"]
    gaps_list = data["gaps_list"]
    matrix = data["matrix"]
    nphi_DOS_list = data["eta_DOS"]
    DOS_list = data["DOS_list"]
    tr_DOS_list = data["tr_DOS"]
    size = (7.5, 9)
    fig1 = plt.figure(figsize=size, dpi=100)
    fig1.canvas.manager.set_window_title("Butterfly Spectrum")
    transparent = True
    colors1 = plt.cm.gist_rainbow(np.linspace(0.75, 1, 10)[::-1])
    colors2 = plt.cm.seismic([0.5])
    colors2[:, -1] = 0
    colors3 = plt.cm.gist_rainbow(np.linspace(0.0, 0.5, 10))
    colors = np.vstack((colors1, colors2, colors3))
    cmap = mcolors.LinearSegmentedColormap.from_list("avron", colors, 21)

    ax1 = fig1.add_subplot(111)
    cp = ax1.imshow(matrix.T, origin="lower", cmap=cmap, extent=[0, 1, -0.01827895998495045, 2.4], aspect="auto", vmin=-10, vmax=10)
    cbar = plt.colorbar(cp)
    cbar.set_label("σ$_{xy}$ (e$^2$/h)", fontsize=13, fontfamily="Arial")
    tick_locs = np.linspace(-10, 10, 2 * 21 + 1)[1::2]
    cbar_tick_label = np.arange(-10, 10 + 1)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(cbar_tick_label)
    ax1.set_ylabel("Energy (eV)", fontsize=16, fontfamily="Arial")
    ax1.set_xlabel("p/q", fontsize=16, fontfamily="Arial")
    ax1.set_xticks(np.arange(0, 1.1, 0.1))
    ax1.set_yticks(np.arange(0, 2.2, 0.5))
    ax1.tick_params(axis="both", direction="out", which="both", labelsize=13)

    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams["font.family"] = "Arial"
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("$%g$"))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%g$"))

    fig2 = plt.figure(figsize=size, dpi=100)
    fig2.canvas.manager.set_window_title("Butterfly Spectrum - Scatter")
    ax2 = fig2.add_subplot(111)

    sc = ax2.scatter(eta_list, E_list, c=clist, cmap=cmap, s=40, marker=".", vmin=-10, vmax=10, linewidths=0)

    cbar2 = plt.colorbar(sc)
    cbar2.set_label("σ$_{xy}$ (e$^2$/h)", fontsize=16, fontfamily="Arial")
    cbar2.set_ticks(tick_locs)
    ax2.set_ylabel("Energy (eV)", fontsize=16, fontfamily="Arial")
    ax2.set_xlabel("p/q", fontsize=16, fontfamily="Arial")
    ax2.set_xticks(np.arange(0, 1.1, 0.1))
    ax2.set_yticks(np.arange(0, 2.2, 0.5))
    ax2.tick_params(axis="both", direction="out", which="both", labelsize=13)

    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams["font.family"] = "Arial"
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("$%g$"))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%g$"))

    fig3 = plt.figure(figsize=size, dpi=100)
    fig3.canvas.manager.set_window_title("wannier")
    ax3 = fig3.add_subplot(111)
    ax3.set_ylabel("Energy (eV)", fontsize=16, fontfamily="Arial")
    ax3.set_xlabel("p/q", fontsize=16, fontfamily="Arial")
    ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter("$%g$"))
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%g$"))

    nphi_DOS_list = list(np.concatenate(nphi_DOS_list).ravel())
    DOS_list = list(np.concatenate(DOS_list).ravel())
    gaps_list = list(np.concatenate(gaps_list).ravel())
    tr_DOS_list = list(np.concatenate(tr_DOS_list).ravel())
    sc3 = ax3.scatter(nphi_DOS_list, DOS_list, s=[60 * i for i in gaps_list], c=tr_DOS_list, cmap=cmap, linewidths=0, vmin=-10, vmax=10)

    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams["font.family"] = "Arial"
    ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter("$%g$"))
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter("$%g$"))
    ax3.set_xticks(np.arange(0, 1.1, 0.1))
    ax3.set_yticks(np.arange(0, 1.1, 0.1))

    cbar3 = plt.colorbar(sc3)
    cbar3.set_label("σ$_{xy}$ (e$^2$/h)", fontsize=13, fontfamily="Arial")
    tick_locs = np.linspace(-10, 10, 2 * 21 + 1)[1::2]
    cbar_tick_label = np.arange(-10, 10 + 1)
    cbar3.set_ticks(tick_locs)
    cbar3.set_ticklabels(cbar_tick_label)

    plt.show()
    return None


def save_data(
    data,
    n_levels,
    qmax,
    file,
    fileMatrix,
    file_plot_Matrix_Gnu,
    fileLandauLevels_valenceBand,
    fileLandauLevels_conductionBand,
    filegnu,
):
    E_list = data["E_list"]
    E_list_orig = data["E_list_orig"]
    eta_list = data["eta_list"]
    clist = data["chern_list"]
    tlist = data["tr_list"]
    klist = data["kj_list"]
    gap_list = data["gaps_list"]
    matrix = data["matrix"]
    matt = data["matt"]
    plist = data["p_list"]
    with open(file, "w", newline="") as writefile:
        header = ["eta", "Energy", "chern", "tr", "matrix", f"p/{qmax}"]
        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        for i in tqdm(range(len(plist)), ascii=" #", desc="Save data"):
            for j in range(len(E_list[i])):
                writer.writerow(
                    {
                        "eta": eta_list[i][j],
                        "Energy": E_list[i][j],
                        "chern": clist[i][j],
                        "tr": tlist[i][j],
                        "matrix": matrix[i][j],
                        f"p/{qmax}": f"{plist[i]}/{qmax}",
                    }
                )
            writefile.write("\n")

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
set ylabel 'Energy (eV)' font 'Arial,16'
set grid

#set xrange [0:0.2]
#set yrange [-0.2:1.75]

set tics out
plot '{file}' u 1:2 with points pt 7 ps 0.03 lc rgb 'black' notitle 'HofstadterButterfly_{matt}' ,\
#plot for [i=2:{n_levels}] "{fileLandauLevels_conductionBand}" using 1:i with lines lw 0.75 lc rgb 'black' notitle
pause -1
        """
        )
    subprocess.run(["gnuplot", filegnu])


def main():
    qmax = 9
    n_levels = 30
    choice = 0
    matt, alattice, e1, e2, t0, t1, t2, t11, t12, t22 = para(choice)
    # file = f"3band_dataHofstadterButterfly_q_{qmax}_{matt}_final.dat"
    # file = f"3band_dataHofstadterButterfly_q_{qmax}_{matt}_chern_final.dat"
    # file = f"3band_dataHofstadterButterfly_q_{qmax}_{matt}_SOC_PLAMBDA.dat"
    # file = f"2band_dataHofstadterButterfly_q_{qmax}_{matt}_final.dat"
    file = f"1band_dataHofstadterButterfly_q_{qmax}_h0_chern.dat"
    # file = f"1band_dataHofstadterButterfly_q_{qmax}_h11.dat"
    # file = f"1band_dataHofstadterButterfly_q_{qmax}_h22.dat"

    # fileMatrix = f"1band_Matrix_q_{qmax}_h0.dat"
    fileMatrix = f"3band_Matrix_q_{qmax}.dat"
    file_plot_Matrix_Gnu = f"Matrix_q_{qmax}.gnuplot"

    fileLandauLevels_conductionBand = f"LandauLevels_q={qmax}_{matt}_c_band.dat"
    fileLandauLevels_valenceBand = f"LandauLevels_q={qmax}_{matt}_v_band.dat"

    filegnu = f"plotHofstadterButterfly_q={qmax}_{matt}_landaulevel.gnuplot"

    print("file data: ", file)
    print("file gnuplot: ", filegnu)
    print("file landau: ", fileLandauLevels_conductionBand, "&", fileLandauLevels_valenceBand)
    print("file Matrix: ", fileMatrix)
    print("file Matrix GNU: ", file_plot_Matrix_Gnu)
    LandauLevels(fileLandauLevels_conductionBand, fileLandauLevels_valenceBand, qmax, n_levels, choice)
    data = process(choice, qmax)
    plotPython(data)
    save_data(
        data,
        n_levels,
        qmax,
        file,
        fileMatrix,
        file_plot_Matrix_Gnu,
        fileLandauLevels_valenceBand,
        fileLandauLevels_conductionBand,
        filegnu,
    )


if __name__ == "__main__":
    main()
