import sturm_liouville
import numpy as np
import matplotlib.pyplot as plt


def potential_1(x):
    return 700 * (0.5 - np.abs(x - 0.5))


def potential_2(x):
    return 800 * np.power(np.sin(np.pi * x), 2)


def potential_3(x):
    return 700 * x * (1 - x)


def potential_4(x):
    return 700 * (1 - np.cos(8 * np.pi * x))


if __name__ == '__main__':
    potentials = [potential_1, potential_2, potential_3, potential_4]

    fig, ax = plt.subplots(1, len(potentials))
    for i, pot in enumerate(potentials):
        prob = sturm_liouville.SturmLiouvilleProblem(pot, neumann_final=False)
        x, v, vec = prob.modes()
        ax[i].plot(x, -pot(x), label="Potential")
        ax_twin = ax[i].twinx()
        for j in range(6):
            ax_twin.plot(x, 50 * np.power(np.abs(vec[:, j]), 2) - 0.0075 * np.real(v[j]),
                         label=f"Mode {i}: Energy level: {np.real(v[j]):.2}")
        ax[i].legend()
    plt.show()
