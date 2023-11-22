import task14
import numpy as np
import matplotlib.pyplot as plt


def van_der_pol(t, x, mu=100.0):
    return np.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])


if __name__ == '__main__':
    y0s = np.array([[-0.5, 0.0], [0.5, 0.0], [0.25, -0.1], [-0.25, 0.1], [2.0, 0.0]])
    mus = np.linspace(0.5, 10.0, 2)
    t1 = 2 * mus
    tol = 1e-5
    t0 = 0.0
    fig, ax = plt.subplots(1, len(mus), figsize=(15, 5))
    for i, mu in enumerate(mus):
        prob = task14.RK34Adaptive(lambda a, b: van_der_pol(a, b, mu), y0s[-1], t0, t1[-1], tol)
        t, y, h, err = prob.solve()
        ax[i].plot(t, y[:, 0], 'b', label='y1')
        ax[i].plot(t, y[:, 1], 'r', label='y2')
        ax[i].set_title(r'$\mu = {}$'.format(mu))
        ax[i].set_xlabel(r'$t$')
        ax[i].set_ylabel(r'$y$')
        ax[i].legend()
    plt.show()
    plt.clf()
    fig = plt.figure(figsize=(15, 5))
    for i, y0 in enumerate(y0s):
        prob = task14.RK34Adaptive(lambda a, b: van_der_pol(a, b, 100.0), y0, t0, 200.0, tol)
        t, y, h, err = prob.solve()
        plt.plot(y[:, 0], y[:, 1], label='y0 = {}'.format(y0))
    plt.title(r'Phase portrait ($\mu = 100$)')
    plt.xlabel(r'$y_1$')
    plt.ylabel(r'$y_2$')
    plt.legend()
    plt.show()




