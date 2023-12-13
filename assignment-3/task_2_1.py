import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import integrator
import custom_anim


class LaxWenIntegrator(integrator.Integrator):
    def step(self, i):
        return self.f @ self.y[i - 1, :]

    def error_est(self):
        return 0.0

a = -1.0
m = 1.0
N = 19
col = np.zeros(N)
col[0] = 1 - a*a*m*m
col[1] = (a*m/2) * (1 + a*m)
col[-1] = -(a*m/2) * (1 - a*m)
C = sp.linalg.circulant(col)
x = np.linspace(0, 1, N+1)
x = x[1:]
dx = x[1] - x[0]
dt = m * dx
y0 = np.exp(-100 * (x - 0.5) ** 2)
prob = LaxWenIntegrator(C, y0, t0=0, t1=5)
prob.apply_discretization_scheme(dt)
t, y, _ = prob.solve()
rms = (1/np.sqrt(N)) * np.linalg.norm(y, axis=1)
plt.plot(t, rms)
plt.show()



custom_anim.make_2danimation(x,y,t,
                             f'Advection(a = {a}, $\mu = {m}$)\nusing Lax-Wendroff scheme',1)