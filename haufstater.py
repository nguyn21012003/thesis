import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import subprocess


np.set_printoptions(precision=10, linewidth=1200, edgeitems=24)


def H(t, p, q, kx, ky):

    alpha = 1 * p / q

    M = np.zeros([q, q], dtype=complex)

    for i in range(0, q):

        M[i, i] = 2 * t * np.cos(2 * np.pi * alpha * i - ky)

        if i == q - 1:
            M[i, i - 1] = t
        elif i == 0:
            M[i, i + 1] = t
        else:
            M[i, i - 1] = t
            M[i, i + 1] = t

    # Bloch conditions
    if q == 2:
        M[0, q - 1] = t + t * np.exp(-q * 1.0j * kx)
        M[q - 1, 0] = t + t * np.exp(q * 1.0j * kx)
    else:
        M[0, q - 1] = t * np.exp(-q * 1.0j * kx)
        M[q - 1, 0] = t * np.exp(q * 1.0j * kx)

    # print(M, "\n")
    # m = 1 thì là phi{2} + phi 1 nhaan e^{q} + phi_{1} = phi_{1} phi_{m+-q} = e{+- q} nhan phi_{m}

    # m = 2 thì là e^{q} nhan phi2 + phi_{1} + phi_{2} = phi_{2} ## phi{3} = e^{q} nhan phi2

    # ( 1 + e^{q} ) phi_{1} + phi_{2} = phi_{1}
    # ( 1 + e^{q} ) phi_{2} + phi_{1} = phi_{2}
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # print(M)
    return M


def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


def plot_butterfly(q_max):
    t = -1
    with open("butterfly.csv", mode="w") as file:
        header = ["alpha", "epsilon"]
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()

        for p in tqdm(range(1, q_max + 1)):

            if gcd(p, q_max) == 1:

                alpha = p / q_max
                y = np.zeros(q_max)
                y[:] = alpha

                x1 = np.linalg.eigvalsh(H(t, p, q_max, kx=4 * np.pi / 3, ky=0))
                # x2 = np.linalg.eigvalsh(H(p, q, kx=np.pi / q, ky=np.pi / q))
                # print(x1)

                for i in range(len(x1)):
                    writer.writerow({"alpha": alpha, "epsilon": x1[i]})

                # for i in range(len(x1)):
                #    plt.plot([x1[i], x2[i]], y[:2], "-", c="red", markersize=0.1)

                # plt.plot(y, x1, ".", c="red", markersize=1)
                # plt.plot(x2, y, "o", c="blue", markersize=0.1)
    with open("GNUPLOT1bandSquare.gp", "w") as gnuplotfile:
        gnuplotfile.write(
            f"""
set datafile separator ','
plot 'butterfly.csv' u 1:2 with points pt 7 ps 0.2 lc rgb 'red' title '1 band q = {q_max}'
pause -1
            """
        )
    subprocess.run(["gnuplot", "GNUPLOT1bandSquare.gp"])

    # plt.xlabel(r"$\epsilon$", fontsize=15)
    # plt.ylabel(r"$\alpha$", fontsize=15)
    # plt.title(r"$q=1-$" + str(q_max))
    # plt.show()

    return


q_max = 151


plot_butterfly(q_max)
