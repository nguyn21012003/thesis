def chern(p, q):
    sr_list, tr_list, kj_list = [], [], []
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
