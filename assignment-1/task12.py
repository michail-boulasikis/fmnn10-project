import integrator
import numpy as np
import plots


class RK34Integrator(integrator.Integrator):
    def __init__(self, f, y0, t0=0.0, t1=1.0):
        super().__init__(None, y0, t0, t1)
        self.f = f
        self.local_error = None

    def apply_discretization_scheme(self, h_or_n):
        super().apply_discretization_scheme(h_or_n)
        self.local_error = np.zeros_like(self.y)

    def error_est(self):
        return self.local_error

    def step(self, i):
        ynp1 = self.y[i - 1] + (self.h / 6) * (
            (Y1 := self.f(self.t[i - 1], self.y[i - 1])) +
            2 * (Y2 := self.f(self.t[i - 1] + self.h / 2, self.y[i - 1] + (self.h / 2) * Y1)) +
            2 * (Y3 := self.f(self.t[i - 1] + self.h / 2, self.y[i - 1] + (self.h / 2) * Y2)) +
            (Y4 := self.f(self.t[i], self.y[i - 1] + self.h * Y3))
        )
        Z3 = self.f(self.t[i - 1] + self.h, self.y[i - 1] - self.h * Y1 + 2 * self.h * Y2)
        self.local_error[i] = (self.h / 6) * (2*Y2 + Z3 - 2*Y3 - Y4)
        return ynp1


# y' = C*y
C = 1.0
# y(0) = 1
y0 = np.array([1.0])
prob = RK34Integrator(lambda t, y: C * y, y0, t0=0.0, t1=10.0)
plots.error_plot(prob, order=4.0)
