import numpy as np

import integrator
import matrices
import matplotlib.pyplot as plt

if __name__ == '__main__':
    t0 = 0
    t1 = 1
    N = 39
    m = 0.49 #Desired
    x = np.linspace(0, 1, N+2)
    dx = x[1] - x[0]
    dt_desired = m * dx**2 #desired
    M = int(1+ (t1-t0)/dt_desired)
    dt = (t1-t0)/(M-1) #Actual value to get an equidistant grid
    mu = dt*dx**-2
    y0 = x * (x - 1)
    y0 = y0[1:-1]
    
    print("------ Diffusion equation ------")
    print("Explicit Euler method")
    print(f"Integrated in the time interval [{t0},{t1}]")
    print("Parameters:")
    print(f"N = {N} ; M = {M}")
    print(f"dt = {dt:.4g} ; dx = {dx:.4g}; mu = {mu:.8g}")
    print('--------------------------------')
    
    prob = integrator.EulerIntegrator(matrices.T(N, dx), y0,t0,t1)
    #prob.apply_discretization_scheme(dt)
    prob.apply_discretization_scheme(M)
    t, y, _ = prob.solve()
    y_padded = np.concatenate((np.zeros((*t.shape, 1)), y, np.zeros((*t.shape, 1))), axis=1)
    fig, ax = plt.subplots(subplot_kw={"projection" : "3d"})
    T, X = np.meshgrid(x, t)
    surf = ax.plot_surface(T, X, y_padded)
    plt.xlabel("x")
    plt.ylabel('t')
    plt.show()