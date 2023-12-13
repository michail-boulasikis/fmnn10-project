import matrices
import numpy as np
import custom_anim
import matplotlib.pyplot as plt


def lax_wendroff(un, Tdx, Sdx, dt):
    return un - (dt * un * (Sdx @ un)) + ((dt * dt / 2) * un * (2 * (Sdx @ un) * (Sdx @ un) + Tdx @ un))


def modified_lax_wendroff(un, Tdx, Sdx, dt, inverse, d=0.1):
    return inverse @ (lax_wendroff(un, Tdx, Sdx, dt) + (d * dt * un / 2))


def solve_burger(u0, Tdx, Sdx, dt, t0=0, t1=1, d=0.1):
    Nt = int((t1 - t0) / dt)
    u = np.zeros((Nt, len(u0)))
    inverse = np.linalg.inv(np.eye(N) - (d * dt / 2) * Tdx)
    u[0, :] = u0
    for i in range(1, Nt):
        u[i, :] = modified_lax_wendroff(u[i - 1, :], Tdx, Sdx, dt, inverse, d)
    return np.linspace(t0, t1, Nt), u


N = 250
d = 0.1
x = np.linspace(0, 1, N + 1)
x = x[1:]
dx = x[1] - x[0]
m = 0.5
dt = m * dx

Tdx = matrices.T(N, dx, circulant=True)
Sdx = matrices.S(N, dx, 'fwd', circulant=True)
u0 = np.exp(-100 * (x - 0.5) ** 2)

t, u = solve_burger(u0, Tdx, Sdx, dt, t0=0, t1=2, d=d)
print(f"x.shape = {x.shape}, t.shape = {t.shape}, u,shape = {u.shape}")
custom_anim.make_2danimation(x, u, t, title=f"Burger Equation with $d = {d}$")
