import casadi as ca


def ca_inv22(mat: ca.SX) -> ca.SX:
    m1, m2 = mat[0, 0], mat[0, 1]
    m3, m4 = mat[1, 0], mat[1, 1]
    inv_det = 1.0 / (m1 * m4 - m2 * m3)
    return ca.blockcat(m4, -m2, -m3, m1) * inv_det


def is_ca_arr(x) -> bool:
    return isinstance(x, (ca.SX, ca.MX, ca.DM))
