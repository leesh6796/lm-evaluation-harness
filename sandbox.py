import torch

if __name__ == "__main__":
    # a = torch.tensor([1, 2, 3, 4, -1, -2, -3, -4])
    # b = a - torch.zeros(a.shape)
    # print(a)
    # print(b)
    # print(a is b)

    # E = 4
    # M = 3
    # bias = 2 ** (E - 1) - 1
    # # min_x = float((2 ** (-M)) * (2 ** (1 - bias)))
    # min_x = float(2 ** (1 - bias - M))
    # # (1 + (1 - 2^(-M))) * 2 ** (2^E - 2 - bias)을 정리한 것.
    # max_x = float((2 - 2 ** (-M)) * (2 ** (bias)))
    # print(min_x, max_x)

    # import torch

    # torch to cupy
    E_16 = 5
    M_16 = 10
    E_8 = 4
    M_8 = 3
    bias_16 = 2 ** (E_16 - 1) - 1
    bias_x = 2 ** (E_8 - 1) - 1
    mask_E16 = ((1 << E_16) - 1) << M_16
    mask_sign = 1 << (E_16 + M_16)
    mask_frac = (1 << M_16) - 1
    mask_frac_Mx = (2**M_8 - 1) << (M_16 - M_8)
    x = torch.rand(3, dtype=torch.float16).cuda().view(torch.int16)
    print(x.view(torch.float16))

    sign = x & mask_sign
    exp = ((x & mask_E16) >> M_16) - bias_16
    frac = x & mask_frac
    print(sign)
    print(exp)
    print(frac)
    exp[exp < -9] = -bias_x  # 표현 범위 넘어가면 0
    frac[exp < -9] = 0
    exp[exp > 7] = 16
    frac[exp > 7] = 0
    print(exp)
    print(frac)
    # denorm_idx = x[(-9 <= exp) & (exp < -6)]
    # dshift = -6 - exp[(-9 <= exp) & (exp < -6)]
    dshift = -6 - exp
    print(dshift)
    frac[(-9 <= exp) & (exp < -6)] >>= dshift[(-9 <= exp) & (exp < -6)]
    print(frac)
    frac[(-9 <= exp) & (exp < -6)] <<= dshift[(-9 <= exp) & (exp < -6)]
    print(frac)
    frac = frac & mask_frac_Mx
    exp += bias_16
    exp <<= M_16
    x = sign | exp | frac
    x = x.view(torch.float16)

    print(x)
    exit(-1)

    x = a + add_for_round
    print(a)
    sign_and_frac = x & (mask_sign + mask_frac_Mx)
    print(a)

    # x[x[(x & mask_frac_Mx) != 0]] += 1 << M_16  # frac이 전부 1이면 지수에 1 더한다.
    x += add_for_round
    E = (x >> M_16) & mask_E16
    x = x & (mask_sign + mask_frac_Mx)
    E = E - bias_16 + bias_x
    E[E < 9] = 0
