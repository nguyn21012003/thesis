import csv
import numpy as np
from tqdm import tqdm

h = 6.626e-34
e = 1.6e-19


def hall_resistance(B, n_e):
    R = h / (e**2)
    rho_list = []
    for B_val in tqdm(B):
        nu = np.round(n_e * h / (e * B_val))
        print(nu)
        if nu >= 1:
            rho_xy = R / (nu * 1000)
            rho_list.append([B_val, rho_xy])
    return rho_list


def saveFile(data):
    with open("hall_resistance.csv", "w", newline="") as f:
        header = ["B", "R_Hall"]
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for B_val, R_Hall in data:
            writer.writerow({"B": B_val, "R_Hall": R_Hall})


def main():
    B = np.arange(0.1, 10, 0.2)
    n_e = 2.8e15
    data = hall_resistance(B, n_e)
    saveFile(data)


if __name__ == "__main__":
    main()
