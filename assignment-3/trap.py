import integrator
import numpy as np
from scipy.linalg import expm


class Trapezoidal(integrator.Integrator):
    """Only for equation of the form u'(t) = Mu(t)
    """

    def __init__(self, M, y0, t0=0, t1=1):
        super().__init__(None, y0, t0, t1)
        self.M = M

    def apply_discretization_scheme(self, h_or_n):
        super().apply_discretization_scheme(h_or_n)
        self.A = self.h / 2 * self.M

    def step(self, i):
        """U_(n+1) = (I-A)⁻¹(I+A)U_n
        """
        I = np.eye(np.shape(self.A)[0])
        b = (I + self.A) @ self.y[i - 1]
        ynp1 = np.linalg.solve(I - self.A, b)
        return ynp1

    def true_err(self):
        y_tru = np.zeros_like(self.y)
        for i, time in enumerate(self.t):
            y_tru[i] = expm(self.M * time) @ self.y0
        self.true_err = y_tru - self.y


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    M = -0.5 * np.eye(2)
    y0 = np.array([-1, 2])
    trap = Trapezoidal(M, y0, 0, 100)
    trap.apply_discretization_scheme(10)
    trap.solve()
    trap.true_err()
