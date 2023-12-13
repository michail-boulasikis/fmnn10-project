import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

def make_2danimation(x,y,t,title = '',speed = 1):
    """

    Args:
        x (_type_): _description_
        y (_type_): _description_
        t (_type_): _description_
        speed (int, optional): Speed multiplier. Defaults to 1(real time,
        assuming t is in second).

    Returns:
        _type_: _description_
    """
    fig = plt.figure()
    xmin, xmax = np.min(x) , np.max(x)
    ymin , ymax = np.min(y) , np.max(y)
    ax = plt.axes(xlim=(xmin,xmax),
                  ylim=(ymin - 0.05*np.abs(ymin), ymax + 0.05*np.abs(ymax)))
    line,  = ax.plot([], [], lw=2) 
    text = ax.text(0.1,0.9,'',fontsize = 14,color = 'r')
    
    def init():
        line.set_data([], [])
        text.set_text('')
        ax.set(xlabel = 'x',
               ylabel = 'y',
               title = title)
        
        return line, text

    def animate(i):
        X = x
        Y = y[i, :]
        line.set_data(X, Y)
        text.set_text(f't = {t[i]:.2f}')
        
        return line, text,
    
    t_int = t[1]- t[0]
    interval = int(t_int*1000/speed)
    print(interval)
    animation = anim.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(t),
                                   interval=interval,
                                   blit=True)
    plt.show()
