import numpy as np
import matrices
import trap
import matplotlib.pyplot as plt


if __name__ == '__main__':
    N = 499
    m = 10.0
    x = np.linspace(0, 1, N+2)
    dx = x[1] - x[0]
    print(f"dx = {dx}")
    y0 = x * (x - 1)
    y0 = y0[1:-1]
    prob = trap.Trapezoidal(matrices.T(N, dx), y0)
    dt = m * dx**2
    prob.apply_discretization_scheme(dt)
    t, y, _ = prob.solve()
    y_padded = np.concatenate((np.zeros((*t.shape, 1)), y, np.zeros((*t.shape, 1))), axis=1)
    fig, ax = plt.subplots(subplot_kw={"projection" : "3d"})
    T, X = np.meshgrid(x, t)
    surf = ax.plot_surface(T, X, y_padded)
    plt.show()