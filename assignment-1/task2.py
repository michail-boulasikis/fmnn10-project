import integrator
from task14 import RK34Adaptive
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


def LVdiff(t, u, a=3, b=9, c=15, d=15):
    """Lotka-Voltera differential equation, in the form f'(t,u)

    # Args:
        t (_type_): time, unused
        u (_type_): 2-vector u = [x,y]
        a (int, optional): reproduction rate of the prey. Defaults to 3.
        b (int, optional): Encounter death rate of the prey. Defaults to 9.
        c (int, optional): Encounter reproduction rate of the predator. Defaults to 15.
        d (int, optional): Death rate of the predator. Defaults to 15.

    # Returns:
        _type_: f'(t,u) = [x',y'] as a numpy array
    """

    x = u[0]
    y = u[1]
    dudt = np.zeros(2)
    dudt[0] = a * x - b * x * y
    dudt[1] = c * x * y - d * y
    return dudt


def H(x, y, a=3, b=9, c=15, d=15):
    return c * x + b * y - d * np.log(x) - a * np.log(y)


def testLV():
    u = [5, 5]
    print(LVdiff(0, u))


y0 = np.array([1, 1])
task = RK34Adaptive(LVdiff, y0, 0, 5, tol=1e-5)
t, u, h, err = task.solve()
x = u[:, 0]
y = u[:, 1]

### First plot
fig1, ax1 = plt.subplots()
fig1.set_figheight(2)
fig1.set_figwidth(3)
ax1.plot(t, u, label=['Rabbit', 'Fox'])
ax1.set(ylabel='Population',
        xlabel='t')
fig1.legend(loc='upper left',
            ncols=2,
            bbox_to_anchor=(0.2, 1.05, 0., 0.))


# fig1.savefig('LotkaVoltTime.pdf',
#            bbox_inches = 'tight')

###Deviation and energy
def make_deviation_plot(tol_list):
    #####Deviation plot
    fig2, ax2 = plt.subplots()
    for tol in tol_list:
        y0 = np.array([1, 1])
        task = RK34Adaptive(LVdiff, y0, 0, 1000, tol=tol)
        t, u, _, _ = task.solve()
        x = u[:, 0]
        y = u[:, 1]
        H_val = H(x, y)
        ax2.plot(t, np.abs(H_val / H_val[0] - 1), label=tol)
    ax2.set(yscale='linear',
            xscale='log',
            title='Relative deviation over time',
            xlabel='t',
            ylabel='|H(x,y)/H(0)-1|')
    ax2.legend(loc='lower left',
               bbox_to_anchor=(1, 0))
    fig2.set_figheight(10)
    fig2.set_figwidth(10)
    return fig2, ax2


fig2, ax2 = make_deviation_plot([1e-5, 1e-6, 1e-7, 1e-8])

###XY plot
fig3, ax3 = plt.subplots()
ax3.plot(x, y)
ax3.set(title='Phase plot',
        xlabel='Rabbit population',
        ylabel='Fox population')
fig3.set_figheight(2)
fig3.set_figwidth(3)

##Misc
w = np.linspace(0.01, 10, 400)
periodogram = signal.lombscargle(t, x, w)

fig4, ax4 = plt.subplots()
ax4.plot(w / (2 * np.pi), periodogram)

plt.show(block=True)
