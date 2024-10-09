import casadi as ca
import casadi_f16.f16
import ipdb
import numpy as np


def main():
    f16 = casadi_f16.f16.F16()

    x_trim, u_trim = f16.trim_state(), f16.trim_control()

    dx_ = ca.MX.sym("dx", f16.NX)
    du_ = ca.MX.sym("du", f16.NU)
    xdot_ = f16.xdot(x_trim + dx_, u_trim + du_)

    # Find closest x, u such that xdot is zero.
    cost_ = ca.sumsqr(dx_) + ca.sumsqr(du_)

    # We don't care about xdot in PN direction.
    xdot_[f16.PN] = 0.0
    g_ = xdot_

    nlp = {"x": ca.vertcat(dx_, du_), "f": cost_, "g": g_}

    lbg, ubg = [0], [0]

    opts = {"print_time": 0, "snopt": {"Major print level": 0, "Minor print level": 0, "Timing level": 0}}
    s = ca.nlpsol("solver", "snopt", nlp, opts)

    trim_state = f16.trim_state()
    trim_control = f16.trim_control()
    # trim_state[f16.H] = 500.0
    ig = np.concatenate([trim_state, trim_control], axis=0)
    result = s(x0=ig, lbg=lbg, ubg=ubg)
    dx_opt = np.array(result["x"][: f16.NX]).squeeze()
    du_opt = np.array(result["x"][f16.NX :]).squeeze()

    print("dx: ", dx_opt)
    print("du: ", du_opt)

    x_opt = trim_state + dx_opt
    u_opt = trim_control + du_opt

    def fmt(arr):
        return ", ".join(["{:.10e}".format(n) for n in arr])

    print("x: ", fmt(x_opt))
    print("u: ", fmt(u_opt))

    xdot_opt = np.array(f16.xdot(x_opt, u_opt)).squeeze()
    print(xdot_opt)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
