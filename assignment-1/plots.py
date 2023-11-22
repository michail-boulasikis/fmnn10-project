import matplotlib.pyplot as plt
import integrator
import numpy as np


def error_plot(problem, order=1.0):
    steps, errors, _ = integrator.error_data_last(problem)
    plt.loglog(steps, errors, label="RK4")
    plt.loglog(steps, 20 * np.power(steps, order), label="h^4", linestyle="--")
    plt.legend([f"Error", f"h^{order}"])
    plt.title(f"{problem.__class__.__name__} Error vs Step Size")
    plt.xlabel("Step Size")
    plt.ylabel("Error")
    plt.show()
