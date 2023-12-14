import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import integrator
import custom_anim
import pandas as pd
import seaborn as sns


class LaxWenIntegrator(integrator.Integrator):
    def step(self, i):
        return self.f @ self.y[i - 1, :]

    def error_est(self):
        return 0.0

t0 = 0
t1 = 5
a = -1.0
m = 1
N = 49
col = np.zeros(N)
col[0] = 1 - a*a*m*m
col[1] = (a*m/2) * (1 + a*m)
col[-1] = -(a*m/2) * (1 - a*m)
C = sp.linalg.circulant(col)
x = np.linspace(0, 1, N+1)
x = x[1:]
dx = x[1] - x[0]
dt = m * dx
y0 = np.exp(-100 * (x - 0.5) ** 2)
prob = LaxWenIntegrator(C, y0, t0=t0, t1=t1)
prob.apply_discretization_scheme(dt)
t, y, _ = prob.solve()
M = len(t)

####Plotting####
print("--- Advection equation u_t + au_x = 0 ---")
print(f"a = {a}")
print(f'Lax-Wendroff scheme integrated for t in [{t[0]:.3g},{t[-1]:.3g}]')
print("Parameters: ")
print(f"N = {N} ; M = {len(t)} ; mu = {m}")
print(f"dt = {dt:.4g} ; dx = {dx:.4g}")



ax , fig = plt.subplots(figsize = (3,3))
rms = (1/np.sqrt(N)) * np.linalg.norm(y, axis=1)
plt.plot(t, rms, label = f'$\\mu = {m}$')



t0 = 0
t1 = 5
a = -1.0
m = 0.9
N = 49
col = np.zeros(N)
col[0] = 1 - a*a*m*m
col[1] = (a*m/2) * (1 + a*m)
col[-1] = -(a*m/2) * (1 - a*m)
C = sp.linalg.circulant(col)
x = np.linspace(0, 1, N+1)
x = x[1:]
dx = x[1] - x[0]
dt = m * dx
y0 = np.exp(-100 * (x - 0.5) ** 2)
prob = LaxWenIntegrator(C, y0, t0=t0, t1=t1)
prob.apply_discretization_scheme(dt)
t, y, _ = prob.solve()
M = len(t)

####Plotting####
print("--- Advection equation u_t + au_x = 0 ---")
print(f"a = {a}")
print(f'Lax-Wendroff scheme integrated for t in [{t[0]:.3g},{t[-1]:.3g}]')
print("Parameters: ")
print(f"N = {N} ; M = {len(t)} ; mu = {m}")
print(f"dt = {dt:.4g} ; dx = {dx:.4g}")




rms = (1/np.sqrt(N)) * np.linalg.norm(y, axis=1)
plt.plot(t, rms, label = f'$\\mu = {m}$')


t0 = 0
t1 = 5
a = -1.0
m = 0.7
N = 49
col = np.zeros(N)
col[0] = 1 - a*a*m*m
col[1] = (a*m/2) * (1 + a*m)
col[-1] = -(a*m/2) * (1 - a*m)
C = sp.linalg.circulant(col)
x = np.linspace(0, 1, N+1)
x = x[1:]
dx = x[1] - x[0]
dt = m * dx
y0 = np.exp(-100 * (x - 0.5) ** 2)
prob = LaxWenIntegrator(C, y0, t0=t0, t1=t1)
prob.apply_discretization_scheme(dt)
t, y, _ = prob.solve()
M = len(t)

####Plotting####
print("--- Advection equation u_t + au_x = 0 ---")
print(f"a = {a}")
print(f'Lax-Wendroff scheme integrated for t in [{t[0]:.3g},{t[-1]:.3g}]')
print("Parameters: ")
print(f"N = {N} ; M = {len(t)} ; mu = {m}")
print(f"dt = {dt:.4g} ; dx = {dx:.4g}")




rms = (1/np.sqrt(N)) * np.linalg.norm(y, axis=1)
plt.plot(t, rms, label = f'$\\mu = {m}$')




plt.xlabel('t')
plt.ylabel(r'$||u||_{RMS}$')
plt.legend()
plt.show()





def characteristic_heatmap(x,y,t,n_ticks = 5):
    fig , ax = plt.subplots(figsize = (4,4))
    flat_y = y.flatten()
    flat_t = np.repeat(t,np.shape(y)[1])
    flat_x = np.tile(x,np.shape(y)[0])
    print(len(flat_t))
    print(len(flat_y))
    print(len(flat_x))

    df = pd.DataFrame({'t' : flat_t, 'y' : flat_y , 'x' : flat_x})
    df2 = df.pivot(index = 't' , columns='x', values = 'y')
    df2 = df2[::-1] #Reverse
    sns.heatmap(df2)
    
    n_x = len(df2.columns)
    n_y = len(df2.index)

    xtick_pos = np.linspace(0,n_x-1,n_ticks)
    xtick_val = [round(df2.columns[int(xtick_pos[i])],3) for i in range(n_ticks)]
    plt.xticks(xtick_pos, xtick_val)
    

    ytick_pos = np.linspace(0,n_y-1,n_ticks).astype(int)
    ytick_val = [round(df2.index[ytick_pos[i]],2) for i in range(n_ticks)]
    plt.yticks(ytick_pos, ytick_val)
    plt.title(f'Characteristics, $a={a}$, $\\mu = {m}$.')
