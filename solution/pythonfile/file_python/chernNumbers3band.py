import numpy as np
from tqdm import tqdm
from numba import cuda


@cuda.jit
def gpucomp(p, q, qmax, tr_list, r_list):

    r = cuda.grid(1)
    if r > qmax:
        return

    for tr in range(-q // 2, q // 2 + 1):
        for sr in range(-q, q + 1):
            if r == q * sr + p * tr:
                tr_list[r] = tr
                r_list[r] = 1
                return

    return None


def chern(p, q, band):
    qmax = band * q

    tr_gpu = cuda.device_array(qmax + 1, dtype=np.int32)
    r_gpu = cuda.device_array(qmax + 1, dtype=np.int32)

    threads_pb = 128

    blocks = (qmax + 1 + threads_pb - 1) // threads_pb

    gpucomp[blocks, threads_pb](p, q, qmax, tr_gpu, r_gpu)
    tr_list = tr_gpu.copy_to_host()
    valid_val = r_gpu.copy_to_host()

    # tr_list = tr_list[valid_val == 1]
    Chern_list = []
    for i in range(qmax):
        Chern_list.append(tr_list[i + 1] - tr_list[i])

    return Chern_list, tr_list


# def testcase():
#     qmax = 797
#     for p in tqdm(range(1, qmax + 1), desc=f"p iter \n"):
#         if np.gcd(p, qmax) == 1:
#             clist, tlist = chern(p, 2 * qmax, 3)
#             print(clist, tlist)
#
#     return None
#
#
# testcase()
#

# def chern(p, q, band):
#     qmax = band * q
#     sr_list, tr_list = [], []
#     for r in range(qmax + 1):
#         for tr in range(-int(q / 2), int(q / 2) + 1):
#             for sr in range(-q, q + 1):
#                 if r == q * sr + p * tr:
#                     sr_list.append(sr)
#                     tr_list.append(tr)
#
#                     break
#             else:
#                 continue
#             break
#
#     Chern_list = []
#     for i in range(qmax):
#         Chern_list.append(tr_list[i + 1] - tr_list[i])
#
#     return Chern_list, tr_list
