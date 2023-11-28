import numpy as np
import scipy as sp


class SturmLiouvilleProblem(object):
    def __init__(self, q, bcx=np.array([0, 1]), neumann_final=True, N=499):
        self.q = q
        # NOTE: Boundary conditions are assumed to be homogeneous
        self.bcx = bcx
        self.neumann_final = neumann_final
        self.N = N
        self.dx = (bcx[1] - bcx[0]) / (N + 1)
        self.x = np.linspace(bcx[0], bcx[1], N + 2)
        self.eigVal = None
        self.eigVec = None

    def modes(self):
        A = self._build_matrix()
        eigVal, eigVec = sp.linalg.eig(A)
        eigVal = np.real(eigVal)
        idx = eigVal.argsort()[::-1]
        eigVal = eigVal[idx]
        eigVec = eigVec[:, idx]
        self.eigVal = eigVal
        # Append initial values:
        self.eigVec = np.zeros((self.N + 2, self.N))
        self.eigVec[1:-1, :] = eigVec
        self.eigVec[0, :] = 0.0
        self.eigVec[-1, :] = self.eigVec[-2, :] if self.neumann_final else 0.0
        return self.x, self.eigVal, self.eigVec

    def _build_matrix(self):
        col = np.zeros(self.N)
        col[0] = -2
        col[1] = 1
        T = sp.linalg.toeplitz(col)
        T[-1][-1] = T[-1][-1] + (4/3 if self.neumann_final else 0)
        T[-1][-2] = T[-1][-2] - (1/3 if self.neumann_final else 0)
        T *= 1.0 / (self.dx * self.dx)
        return T - np.diag(self.q(self.x[1:-1]))
