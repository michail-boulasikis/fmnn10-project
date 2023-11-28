import sturm_liouville
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys


def potential_1(x):
    return 700 * (0.5 - np.abs(x - 0.5))


def potential_2(x):
    return 800 * np.power(np.sin(np.pi * x), 2)


def potential_3(x):
    return 700 * x * (1 - x)


def potential_4(x):
    return 700 * (1 - np.cos(8 * np.pi * x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='Plots either the wave functions or probability functions of'
                    ' various potentials by solving Schroedinger\'s equation, based on a passed flag',
        epilog='Part of FMNN10'
    )
    parser.add_argument("-w", "--wave", action="store_true", help="Plot wave functions")
    parser.add_argument("-p", "--probability", action="store_true", help="Plot probability functions")
    args = parser.parse_args(sys.argv[1:])
    if not args.wave and not args.probability:
        raise ValueError("Either -w or -p must be passed")
    potentials = [potential_1, potential_2, potential_3, potential_4]
    titles = [
        "$V(x) = 700(0.5 - \\|x - 0.5\\|)}$",
        "$V(x) = 800\\sin^2(\\pi x)$",
        "$V(x) = 700x(1-x)$",
        "$V(x) = 700(1 - \\cos(8\\pi x))$"
    ]

    fig, ax = plt.subplots(1, len(potentials))
    for i, pot in enumerate(potentials):
        prob = sturm_liouville.SturmLiouvilleProblem(pot, neumann_final=False)
        x, v, vec = prob.modes()
        ax[i].plot(x, -pot(x), '--', label="Potential")
        ax[i].set_title(titles[i])
        ax_twin = ax[i].twinx()
        potential_range = np.max(-pot(x)) - np.min(-pot(x))
        for j in range(6):
            nvec = (vec[:, j] * vec[:, j]) if args.probability else vec[:, j]
            normalized = nvec / np.max(np.abs(nvec)) * potential_range / 10
            if args.probability:
                ax_twin.plot(x, normalized + v[j], label=f"Mode {i}: Energy level: {np.real(v[j]):.2}")
            elif args.wave:
                ax_twin.plot(x, normalized + v[j], label=f"Mode {i}: Energy level: {np.real(v[j]):.2}")
            ax_twin.set_yticklabels([])
        ax[i].legend()
    plt.show()
