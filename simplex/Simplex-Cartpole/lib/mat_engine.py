import matlab
import matlab.engine
from numpy import linalg as LA
import matplotlib.pyplot as plt
from collections import deque
from numpy.linalg import pinv
import numpy as np
import cvxpy as cp
import copy


class MatEngine:

    def __init__(self):
        self.engine = None
        self.matlab_engine_launch()
        self.cvx_setup()

    def matlab_engine_launch(self):
        self.engine = matlab.engine.start_matlab()
        self.engine.cd("./mat_files/")
        print("Matlab current working directory is ---->>>", self.engine.pwd())

    def cvx_setup(self):
        self.engine.cd("./cvx/")
        print("Setting up the CVX Toolbox...")
        _ = self.engine.cvx_setup
        print("CVX Toolbox setup done.")
        self.engine.cd("..")

    def feedback_law(self,
                     As: np.ndarray,
                     Bs: np.ndarray,
                     Ak: np.ndarray,
                     Bk: np.ndarray,
                     Sc: np.ndarray,
                     Sd: np.ndarray):
        As = As.reshape(4, 4)
        Bs = Bs.reshape(4, 1)
        Ak = Ak.reshape(4, 4)
        Bk = Bk.reshape(4, 1)
        Sc = Sc.reshape(4, 1)
        Sd = Sd.reshape(4, 1)

        As = matlab.double(As.tolist())
        Bs = matlab.double(Bs.tolist())
        Ak = matlab.double(Ak.tolist())
        Bk = matlab.double(Bk.tolist())
        Sc = matlab.double(Sc.tolist())
        Sd = matlab.double(Sd.tolist())

        # return self.engine.feedback_control(As, Bs, Ak, Bk, Sc, Sd)
        return self.engine.feedback_control2(As, Bs, Ak, Bk, Sc, Sd)

    def feedback_control_cvxpy(self, Ac, Bc, Ak, Bk, sc, sd):
        Ac = Ac.reshape(4, 4)
        Bc = Bc.reshape(4, 1)
        Ak = Ak.reshape(4, 4)
        Bk = Bk.reshape(4, 1)
        sc = sc.reshape(4, 1)
        sd = sd.reshape(4, 1)

        # Constants
        n = 4
        alpha = 0.96

        # Calculating error and its absolute value
        e = sc - sd
        val = np.abs(sc - sd)

        # Define D matrix
        D = np.array([[1 / 0.4, 0, 0, 0],
                      [0, 1 / 4.5, 0, 0],
                      [0, 0, 1 / 0.4, 0],
                      [0, 0, 0, 1 / 4.5]])

        # Define CVXPY variables
        Q = cp.Variable((n, n), symmetric=True)
        R = cp.Variable((1, n))

        # Define CVXPY optimization problem
        objective = cp.Maximize(cp.log_det(Q))
        print(alpha * Q, Q @ Ak.T + R.T @ Bk.T)
        constraints = [cp.bmat([[alpha * Q, Q @ Ak.T + R.T @ Bk.T],
                                [Ak @ Q + Bk @ R, Q]]) >> 0,
                       D @ Q @ D.T - np.eye(4) << 0]

        prob = cp.Problem(objective=objective, constraints=constraints)

        # Solve the problem
        prob.solve()

        # Extract solution
        K = np.array(R.value) @ np.linalg.pinv(Q.value)
        M = Ac + Bc @ K

        # Check stability
        assert np.all(np.linalg.eigvals(M) < 0)

        return K

    # Example usage:
    # Provide appropriate matrices Ac, Bc, Ak, Bk, sc, sd
    # K = feedback_control(Ac, Bc, Ak, Bk, sc, sd)


if __name__ == '__main__':
    As = np.array([[0, 1, 0, 0],
                   [0, 0, -1.42281786576776, 0.182898194776782],
                   [0, 0, 0, 1],
                   [0, 0, 25.1798795199119, 0.385056459685276]])

    Bs = np.array([[0,
                    0.970107410065162,
                    0,
                    -2.04237185222105]])

    Ak = np.array([[1, 0.0100000000000000, 0, 0],
                   [0, 1, -0.0142281786576776, 0.00182898194776782],
                   [0, 0, 1, 0.0100000000000000],
                   [0, 0, 0.251798795199119, 0.996149435403147]])

    Bk = np.array([[0,
                    0.00970107410065163,
                    0,
                    -0.0204237185222105]])

    sd = np.array([[0.234343490000000,
                    0,
                    -0.226448960000000,
                    0]])

    mat = MatEngine()
    K = mat.feedback_law(As, Bs, Ak, Bk, sd)
    print(K)
