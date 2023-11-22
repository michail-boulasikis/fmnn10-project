import integrator
import numpy as np
import plots


class RK4Integrator(integrator.Integrator):
    def __init__(self, f, y0, t0=0.0, t1=1.0, coeff=1.0):
        super().__init__(None, y0, t0, t1)
        self.f = f
        self.coeff = coeff

    def error_est(self):
        return np.exp(self.coeff * self.t.reshape(-1, 1)) - self.y

    def step(self, i):
        return self.y[i - 1] + (self.h / 6) * (
                (Y1 := self.f(self.t[i - 1], self.y[i - 1])) +
                2 * (Y2 := self.f(self.t[i - 1] + self.h / 2, self.y[i - 1] + (self.h / 2) * Y1)) +
                2 * (Y3 := self.f(self.t[i - 1] + self.h / 2, self.y[i - 1] + (self.h / 2) * Y2)) +
                self.f(self.t[i], self.y[i - 1] + self.h * Y3)
        )


# y' = C*y
C = 1.0
# y(0) = 1
y0 = np.array([1.0])
prob = RK4Integrator(lambda t, y: C * y, y0, t0=0.0, t1=10.0, coeff=C)
plots.error_plot(prob, order=4.0)
