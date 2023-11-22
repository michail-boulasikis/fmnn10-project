import task14
import numpy as np
import matplotlib.pyplot as plt


def van_der_pol(t, x, mu=100.0):
    return np.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])


if __name__ == '__main__':
    y0 = np.array([2.0, 0.0])
    mu = np.array([10, 15, 22, 33, 47, 68, 100, 150, 220, 330, 470, 680])
    t0 = np.zeros_like(mu)
    t1 = 0.7 * mu
    tol = 1e-7
    n_steps = np.zeros_like(mu)
    for i in range(len(mu)):
        prob = task14.RK34Adaptive(lambda a, b: van_der_pol(a, b, mu[i]), y0, t0[i], t1[i], tol)
        t, y, h, err = prob.solve()
        n_steps[i] = len(t)
    plt.loglog(mu, n_steps, 'b', label='Number of steps')
    plt.loglog(mu, np.power(mu, 2.0), 'r--', label=r'$\mu^2$')
    plt.title(r'Number of steps vs. $\mu$')
    plt.xlabel(r'$\mu$')
    plt.ylabel('Number of steps')
    plt.legend()
    plt.show()