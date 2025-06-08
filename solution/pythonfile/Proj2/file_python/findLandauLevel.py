import csv

from numpy import pi, sqrt
from parameters import para
from tqdm import tqdm
from condition import gcd


def LandauLevels(fileLandauLevels_con, fileLandauLevels_val, fileLandauLevels_1, fileLandauLevels_2, fileLandauLevels_3, qmax, n_levels, choice):
    matt, alattice, e1, e2, t0, t1, t2, t11, t12, t22 = para(choice)

    with open(fileLandauLevels_2, "w", newline="") as fc:
        conduction = ["alpha_con"] + [f"E_{n}" for n in range(n_levels)]
        write_con = csv.DictWriter(fc, fieldnames=conduction, delimiter=",")
        write_con.writeheader()
        for p_con in tqdm(range(1, qmax + 1), ascii=" #", desc="Landau Levels 1"):
            if gcd(p_con, qmax) == 1:
                alpha_con = p_con / qmax
                row_con = {"alpha_con": alpha_con}
                for n1 in range(n_levels):
                    b1 = sqrt(8 * pi * alpha_con / sqrt(3))
                    E_con = (
                        -0.3e1 / 0.2e1 * b1**2 * t11 * n1
                        - 0.3e1 / 0.2e1 * b1**2 * t22 * n1
                        - 0.3e1 / 0.4e1 * t11 * b1**2
                        - 0.3e1 / 0.4e1 * t22 * b1**2
                        + e2
                        + 3 * t11
                        + 3 * t22
                        - 0.3e1
                        / 0.8e1
                        * sqrt(
                            4 * n1**2 * t11**2 * b1**4
                            - 8 * t22 * n1**2 * t11 * b1**4
                            + 4 * n1**2 * t22**2 * b1**4
                            + 4 * t11**2 * b1**4 * n1
                            - 8 * t11 * b1**4 * t22 * n1
                            + 4 * t22**2 * b1**4 * n1
                            + 5 * t11**2 * b1**4
                            - 10 * t11 * b1**4 * t22
                            + 5 * t22**2 * b1**4
                        )
                    )
                    row_con[f"E_{n1}"] = E_con

                write_con.writerow(row_con)

    v = -1  ### a

    data_list = {"alpha2": [], "E1": [], "E2": [], "E3": []}

    files = {
        "E1": "fileLandauLevels_E1.csv",
        "E2": "fileLandauLevels_E2.csv",
        "E3": "fileLandauLevels_E3.csv",
        "E_F": f"fileFermi_q={qmax}.csv",
    }
    with open(files["E3"], "w", newline="") as f:
        fieldnamess = [f"alpha"] + [f"E1_{level}" for level in range(-1, n_levels)]
        writer = csv.DictWriter(f, fieldnames=fieldnamess, delimiter=",")
        writer.writeheader()
        for p in tqdm(range(1, qmax + 1), ascii=" #", desc="Find roots"):
            if gcd(p, qmax) == 1:
                alpha = p / qmax
                data_list["alpha2"].append(alpha)
                row = {"alpha": alpha}
                b = sqrt(8 * pi * alpha / sqrt(3))  #### sua o day

                # for n in range(-1, n_levels):
                #    u = -3 * (t11 + t22 + t0) * (1 / 2 + n) * b**2 + e1 + 6 * t0 + 6 * t11 + 6 * t22 + 2 * e2  ### b
                #    r = (
                #        ((-216 * t11**2 + (-720 * t22 - 1152 * t0) * t11 - 216 * t22**2 - (1152 * t0) * t22 + 288 * t2**2) * n**2 + (-216 * t11**2 + (-720 * t22 - 1152 * t0) * t11 - 216 * t22**2 - (1152 * t0) * t22 + 288 * t2**2) * n + 18 * t11**2 + (-324 * t22 - 288 * t0) * t11 + 18 * t22**2 - (288 * t0) * t22 + 360 * t2**2) * b**4 / 128
                #        - 3 * ((-e1 - 9 * t0) * t11 + (-e1 - 9 * t0) * t22 - t0 * e2 - 6 * t1**2 + (t11 + t22 + t0) * (-3 * t11 - 3 * t22 - e2)) * (1 / 2 + n) * b**2
                #        - (-e1 - 6 * t0) * (-6 * t11 - 6 * t22 - 2 * e2)
                #        - (-3 * t11 - 3 * t22 - e2) ** 2
                #    )
                #    q = (
                #        -81 * ((t11**2 * t0 + ((10 * t0) * t22 / 3 - (2 * t2**2) / 3) * t11 + t22**2 * t0 - (2 * t2**2) * t22 / 3) * n**2 + (t11**2 * t0 + ((10 * t0) * t22 / 3 - (2 * t2**2) / 3) * t11 + t22**2 * t0 - (2 * t2**2) * t22 / 3) * n - t11**2 * t0 / 12 + ((3 * t0) * t22 / 2 - (3 * t2**2) / 2) * t11 - t22 * (t0 * t22 + 34 * t2**2) / 12) * (1 / 2 + n) * b**6 / 16
                #        + (
                #            ((216 * e1 + 4752 * t0) * t11**2 + ((720 * e1 + 11232 * t0) * t22 + (1152 * t0) * e2 + 1728 * t1**2 - 864 * t2**2) * t11 + (216 * e1 + 4752 * t0) * t22**2 + ((1152 * t0) * e2 + 5184 * t1**2 - 864 * t2**2) * t22 - (288 * t2**2) * e2) * n**2
                #            + ((216 * e1 + 4752 * t0) * t11**2 + ((720 * e1 + 11232 * t0) * t22 + (1152 * t0) * e2 + 1728 * t1**2 - 864 * t2**2) * t11 + (216 * e1 + 4752 * t0) * t22**2 + ((1152 * t0) * e2 + 5184 * t1**2 - 864 * t2**2) * t22 - (288 * t2**2) * e2) * n
                #            + (-18 * e1 + 756 * t0) * t11**2
                #            + ((324 * e1 + 3672 * t0) * t22 + (288 * t0) * e2 + 1296 * t1**2 - 1080 * t2**2) * t11
                #            + (-18 * e1 + 756 * t0) * t22**2
                #            + ((288 * t0) * e2 + 2160 * t1**2 - 1080 * t2**2) * t22
                #            - (360 * t2**2) * e2
                #        )
                #        * b**4
                #        / 128
                #        - 3 * ((-e1 - 9 * t0) * t11 + (-e1 - 9 * t0) * t22 - t0 * e2 - 6 * t1**2) * (-3 * t11 - 3 * t22 - e2) * (1 / 2 + n) * b**2
                #        - (-e1 - 6 * t0) * (-3 * t11 - 3 * t22 - e2) ** 2
                #    )
                #    roots_list = np.roots([v, u, r, q])
                #    row[f"E1_{n}"] = roots_list[2]
                #    data_list["E1"].append(np.real(roots_list[0]))
                #    data_list["E2"].append(np.real(roots_list[1]))
                #    data_list["E3"].append(np.real(roots_list[2]))
                # writer.writerow(row)


#    with open(fileLandauLevels_3, "w", newline="") as fc:
#        for p3 in tqdm(range(1, qmax + 1), ascii=" #", desc="Landau Levels 3"):
#            if gcd(p3, qmax) == 1:
#                alpha3 = p3 / qmax
#                values_3 = [alpha3]
#                for n3 in range(n_levels):
#                    E_3 = t0 * (6 - 8 * pi * sqrt(3) * alpha3 * (n3 + 0.5)) + e1
#                    # E = -t0 * sqrt(2 * pi * sqrt(3) * n * alpha) + e1
#                    values_3.append(E_3)
#
#                fc.write(",".join(map(str, values_3)) + "\n")
