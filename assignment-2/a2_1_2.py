from a2_1_1 import *
from inspect import isfunction

class StaticBeam:
    def __init__(self,
                 q = -50e3 ,
                 L = 10,
                 E = 1.9e11) -> None: 
        if(isfunction(q)):
            self.q = q
        else:
            self.q = lambda x : q*np.ones(len(x))
        self.L =  L 
        self.E = E
        self.M = None
        pass

    
    def I(self,x):
        I = 1e-3*(3-2*np.cos(np.pi*x/self.L)**12)
        self.I = I
        return I
    
    def M2(self,x):
        M2 = self.q(x)
        self.M2 = M2
        return M2

    def u2(self,x):
        return self.M[1:-1]/(self.E*self.I(x))
    
    def solve(self,N):
        bvp_M = BVP_solver(lambda x : self.M2(x),0,self.L,0,0,N)
        bvp_M.solve(N)
        self.M = bvp_M.y
        self.x = bvp_M.x
        bvp_u = BVP_solver(lambda x : self.u2(x) , 0, self.L ,0,0, N)
        bvp_u.solve()
        self.u = bvp_u.y
        
    
if __name__ == '__main__':  
    
    N = 999
    beam = StaticBeam()
    beam.solve(N)
    fig , axs = plt.subplots(figsize = (5,2))
    axs.plot(beam.x,beam.u)
    axs.axvline(beam.x[int((N+1)/2)],linestyle = 'dotted', color = 'red')
    axs.set(xlabel = 'x / m',
               ylabel = 'u(x) / m')
    print(f'Deformation at the middle point: {beam.u[int((N+1)/2)]} m')

            
