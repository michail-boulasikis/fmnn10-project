import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import toeplitz
sns.set_theme(style = 'whitegrid')

class BVP_solver:
    """
    Simple solver for boundary problem of the form y''(x) = f(x,y)
    with Dirichlet boundary conditions y(a) = alpha, y(b) = beta.
    """
    def __init__(self,f,a=0,b=1,alpha = 0, beta = 0,N = 100) -> None:
        """

        Args:
            f (function): Vectorial function f in the BVP y''(x) = f(x).
            Must be called of the form f(x).
            a (float, optional): _description_. Defaults to 0.
            b (float, optional): _description_. Defaults to 1.
            alpha (float, optional): _description_. Defaults to 0.
            beta (float, optional): _description_. Defaults to 0.
            N (int, optional): Number of interior nodes. Defaults to 100.
        """
        self.f = f
        assert a<b , "a should be strictly less than b"
        self.a = a
        self.b = b
        assert isinstance(N,int) , "N should be an integer"
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.is_discretized = False
        pass
    
    def discretize(self,N = None):
        """Create the grid and the T matrix
        """
        if N is None:
            N = self.N
        else:
            self.N = N
        T_col = np.zeros(N)
        T_col[0] = -2
        T_col[1] = 1
        self.T = toeplitz(T_col) #Hermitian/Symmetric on our case
        self.delta_x = (self.b-self.a)/(N+1)
        self.delta_x_m2 = self.delta_x**-2
        self.T *= self.delta_x_m2
        self.x = np.linspace(self.a,self.b,N+2)
        self.fbarx = self.f(self.x[1:-1])
        self.fbarx[0] += -self.alpha*self.delta_x_m2
        self.fbarx[-1] += -self.beta*self.delta_x_m2
        self.is_discretized = True
        pass
    
    def solve(self,N = None):
        #Allow for redefining of N directly here
        if self.is_discretized == False or not(N is None):
            self.discretize(N)
        self.y = np.linalg.solve(self.T,self.fbarx)
        self.y = np.concatenate(([self.alpha],self.y,[self.beta]))
        return self.x , self.y
    
    def error_plot(self,y_true,N_list=[]):
        """
        Args:
            y_true (_type_): _description_
            N_list (_type_,optional): _description_
        """
        N_list = np.array(N_list)
        if(N_list.size == 0):
            N_list = np.arrange(0,10)
            N_list = 2**N_list #Exponential spacing
        delta_x_list = np.zeros(len(N_list))
        rmse_list = np.zeros(len(N_list))
        for i, N in enumerate(N_list):
            x, y = self.solve(N)
            y_tru = y_true(x)
            err = y-y_tru
            rms = np.sqrt(self.delta_x)*np.linalg.norm(err,2)
            delta_x_list[i] = self.delta_x
            rmse_list[i]=rms
        
        
        plt.figure(figsize=(3, 3))
        plt.plot(delta_x_list,rmse_list,label='RMS error')
        plt.plot(delta_x_list,delta_x_list**2,label=r'$x^2$')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Global error plot')
        plt.xlabel(r'$\Delta x$')
        plt.ylabel('RMS discretization error')
        plt.legend()
        
    
            
        
if __name__ == '__main__':
    omega = 3
    freq = 2*np.pi*omega

    def y(x):
        return np.cos(omega*x)

    def f(x):
        return -omega**2*np.cos(omega*x)



    a = -np.pi
    b = np.pi
    alpha = y(a)
    beta = y(b)

    problem = BVP_solver(f,a,b,alpha,beta,N=15)
    problem.solve()

    x = np.linspace(a,b,100)

    plt.figure(figsize=(3,3))
    plt.plot(problem.x,problem.y,label='discretized solution',marker='v')
    plt.plot(x,y(x),label='true solution')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.legend(loc='lower left')


    N_list = np.arange(1,12)
    N_list = 2**N_list
    problem.error_plot(y,N_list)
