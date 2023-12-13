import matrices
import trap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import custom_anim

d = 0.01

a = 1.0
if(a > 0):
    dir = 'bwd'
else:
    dir = 'fwd'
    

N = 250
x = np.linspace(0, 1, N+1)
x = x[1:]
dx = x[1] - x[0]

M = d * matrices.T(N, dx, circulant=True) \
    - a * matrices.S(N, dx,dir,circulant=True)
y0 = np.exp(-100 * (x - 0.5) ** 2)


prob = trap.Trapezoidal(M, y0,0,5)
prob.apply_discretization_scheme(1000)
t, y, _ = prob.solve()

custom_anim.make_2danimation(x,y,t,
                             f'Convection diffusion equation\nd = {d}, a = {a}')
