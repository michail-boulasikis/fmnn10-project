import matrices
import trap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


d = 0.01
a = 1.0
N = 250
x = np.linspace(0, 1, N+1)
x = x[1:]
dx = x[1] - x[0]

M = d * matrices.T(N, dx, circulant=True) - a * matrices.S(N, dx, circulant=True)
y0 = np.exp(-100 * (x - 0.5) ** 2)


prob = trap.Trapezoidal(M, y0)
prob.apply_discretization_scheme(1000)
t, y, _ = prob.solve()

fig = plt.figure()
ax = plt.axes(xlim=(0,1), ylim=(-0.2,1))
line, = ax.plot([], [], lw=2)


def init():
    line.set_data([], [])
    return line,

def animate(i):
    X = x
    Y = y[i, :]
    line.set_data(X, Y)
    return line,


animation = anim.FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=10, blit=True)
plt.show()