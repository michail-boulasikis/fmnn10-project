import integrator
import plots

import numpy as np


class RK34Adaptive(integrator.AdaptiveIntegrator):

    def __init__(self, f, y0, t0=0.0, t1=1.0, tol=1e-6):
        super().__init__(f, y0, t0, t1, tol)

    def step(self, h):
        ynp1 = self.y[-1] + (h / 6) * (
            (Y1 := self.f(self.t[-1], self.y[-1])) +
            2 * (Y2 := self.f(self.t[-1] + h / 2, self.y[-1] + (h / 2) * Y1)) +
            2 * (Y3 := self.f(self.t[-1] + h / 2, self.y[-1] + (h / 2) * Y2)) +
            (Y4 := self.f(self.t[-1] + h, self.y[-1] + h * Y3))
        )
        Z3 = self.f(self.t[-1] + h, self.y[-1] - h * Y1 + 2 * h * Y2)
        self.err.append((h / 6) * (2*Y2 + Z3 - 2*Y3 - Y4))
        return ynp1

    def error_est(self, h):
        pass


if __name__ == '__main__':
    # y' = C*y
    C = 1.0
    # y(0) = 1
    y0 = np.array([1.0])
    tols = [1e-2, 1e-4, 1e-6, 1e-8]
    for tol in tols:
        prob = RK34Adaptive(lambda t, y: C * y, y0, t0=0.0, t1=10.0, tol=tol)
        t, y, h, err = prob.solve()
        plots.plt.semilogy(t, np.abs(err.reshape(-1)), label=f"tol={tol}")
    plots.plt.title("RK34Adaptive Local Error")
    plots.plt.xlabel("t")
    plots.plt.ylabel("Local Error")
    plots.plt.legend()
    plots.plt.show()
