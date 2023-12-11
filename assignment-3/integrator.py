import abc

import numpy as np
import scipy.linalg as linalg


class Integrator(object):
    def __init__(self, f, y0, t0=0.0, t1=1.0):
        # Problem variables
        self.f = f
        self.y0 = y0
        self.t0 = t0
        self.t1 = t1
        # Discretization variables
        self.t = None
        self.h = None
        # Solution variables
        self.y = None
        # Approximation for the error
        self.err = None

    def apply_discretization_scheme(self, h_or_n):
        if (type(h_or_n) == int
                or type(h_or_n) == np.int64 or type(h_or_n) == np.int32
                or type(h_or_n) == np.uint64 or type(h_or_n) == np.uint32):
            if h_or_n <= 1:
                raise ValueError(f"N must be greater than 1.")
            self.t = np.linspace(self.t0, self.t1, h_or_n)
            self.h = self.t[1] - self.t[0]
            self.y = np.zeros((len(self.t), *self.y0.shape))
            self.err = np.zeros_like(self.y)
        elif (type(h_or_n) == float
              or type(h_or_n) == np.float64 or type(h_or_n) == np.float32):
            if h_or_n > 1.0 or h_or_n <= 0.0:
                raise ValueError(f"Step size must be between 0 and 1.")
            self.t = np.arange(self.t0, self.t1, h_or_n)
            self.h = h_or_n
            self.y = np.zeros((len(self.t), *self.y0.shape))
            self.err = np.zeros_like(self.y)
        else:
            raise TypeError(f"Parameter passed to discretization function must be"
                            f" either int or float, got {type(h_or_n)} instead.")

    @abc.abstractmethod
    def step(self, i):
        pass

    @abc.abstractmethod
    def error_est(self):
        pass

    def solve(self):
        if self.t is None:
            raise RuntimeError("Discretization scheme not applied. Apply a discretization scheme before solving.")
        self.y[0] = self.y0
        for i in range(1, len(self.t)):
            self.y[i] = self.step(i)
        self.err = self.error_est()
        return self.t, self.y, self.err


class AdaptiveIntegrator(object):
    def __init__(self, f, y0, t0=0.0, t1=1.0, tol=1e-6):
        # Problem variables
        self.f = f
        self.y0 = y0
        self.t0 = t0
        self.t1 = t1
        self.tol = tol
        self.t = [t0]
        self.h = []
        # Solution variables
        self.y = [y0]
        # Approximation for the error
        self.err = [np.ones_like(y0) * self.tol]

    @abc.abstractmethod
    def step(self, h):
        pass

    @abc.abstractmethod
    def error_est(self, h):
        pass

    def adapt(self, hold, err, errm1):
        return hold * ((self.tol / err) ** 0.1666) * ((self.tol / errm1) ** -0.0833)

    def solve(self):
        t = self.t0
        y = self.y0
        h = (np.abs(self.t1 - self.t0) * self.tol ** 0.25) / (100 * (1 + linalg.norm(y)))
        while t < self.t1:
            if t + h > self.t1:
                h = self.t1 - t
            y1 = self.step(h)
            err = self.err[-1]
            errn = linalg.norm(err)
            t += h
            y = y1
            self.t.append(t)
            self.y.append(y)
            self.h.append(h)
            h = self.adapt(h, errn, linalg.norm(self.err[-2]))
        return np.array(self.t), np.array(self.y), np.array(self.h), np.array(self.err)


class EulerIntegrator(Integrator):
    def step(self, i):
        return self.y[i - 1, :] + self.h * self.f @ self.y[i - 1, :]

    def error_est(self):
        return 0.0


def error_data_last(integrator, _steps=None):
    if issubclass(integrator.__class__, Integrator):
        if _steps is None:
            _steps = np.logspace(-2, 0, 100)
        _errors = np.zeros((len(_steps), *integrator.y0.shape))
        for idx, step in enumerate(_steps):
            integrator.apply_discretization_scheme(step)
            t, y, err = integrator.solve()
            _errors[idx] = np.abs(err[-1])
        return _steps, _errors, np.diff(_errors, axis=0) / np.diff(_steps)[..., np.newaxis]
    else:
        raise NotImplementedError("Only integrators with fixed step size are supported for error_data_last().")
