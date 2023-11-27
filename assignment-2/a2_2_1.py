import sturm_liouville
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    prob = sturm_liouville.SturmLiouvilleProblem(lambda x: np.zeros_like(x))
    x, va, vec = prob.modes()
    for i in range(4):
        plt.plot(x, vec[:, i], label=f"Mode {i}: $\\lambda = {np.real(va[i]):2.2}$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    plt.clf()
    Ns = np.logspace(1, 3, 20, dtype=np.int32)
    errors = np.zeros((3, *Ns.shape))
    actual_va = - np.pi * np.pi * ((2*np.array((0, 1, 2)) + 1) ** 2) / 4
    # actual_va = - np.pi * np.pi * (np.array((0, 1, 2)) + 1) ** 2
    for i, nN in enumerate(Ns):
        prob = sturm_liouville.SturmLiouvilleProblem(lambda x: np.zeros_like(x), N=nN)
        x, va, vec = prob.modes()
        print(f"Actual eigenvalues = {actual_va}, eigenvalues with N = {nN}: {va[0:3]}")
        errors[:, i] = np.abs(np.real(va[0:3]) - actual_va)
    plt.loglog(1/(Ns + 1), errors[0], label="$\\lambda_0$")
    plt.loglog(1/(Ns + 1), errors[1], label="$\\lambda_1$")
    plt.loglog(1/(Ns + 1), errors[2], label="$\\lambda_2$")
    plt.loglog(1/(Ns + 1), 5000/((Ns + 1) * (Ns + 1)), label="Reference line $\\Delta x^2$")
    plt.legend()
    plt.title("Eigenvalue Error vs. $\\Delta x$")
    plt.xlabel("$\\Delta x$")
    plt.ylabel("Error")
    plt.show()
