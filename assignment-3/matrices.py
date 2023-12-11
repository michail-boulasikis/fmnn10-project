"""
Just a file with some helpful matrices that are created when semi discretizing
"""

from scipy.linalg import toeplitz
import numpy as np

def T(n,delta_x,circulant = False):
    """Discretized operator for the second spatial derivative
    d² / dx² using 2nd order central difference
    
    ## Args:
        n (int): dimension of the matrix
        delta_x (float): step size
        circulant (bool, optional): True if periodic BC. Defaults to False.

    ## Returns:
        T_{delta_x}: The discretized operator
    """
    c = np.zeros(n)
    c[0] = -2
    c[1] = 1
    T = toeplitz(c)
    if(circulant):
        T[0,-1] = 1
        T[-1,0] = 1 
    T *= 1/delta_x**2
    return T

def S(n,delta_x, direction = 'bwd',circulant = False):
    """Discretized operator for the second spatial derivative
    d / dx using 1st  order backward or forward difference
    ## Args:
        n (int): dimension of the matrix
        delta_x (float): step size
        direction (str, optional) : Must either be "bwd" or "fwd" for backward
        or forward difference. Defaults to "bwd" 
        circulant (bool, optional): True if periodic BC. Defaults to False.

    ## Returns:
        S_{delta_x}: The discretized operator
    """
    ones = np.ones(n)
    if(direction == 'bwd'):
        S1 = np.diag(ones)
        S2 = np.diag(-ones[1:],k=-1)
        S = S1 + S2
        if(circulant):
            S[0,-1] = -1
    elif(direction == 'fwd'):
        S1 = np.diag(-ones)
        S2 = np.diag(ones[1:],k=1)
        S = S1 + S2
        if(circulant):
            S[-1,0] = 1
    else:
        raise ValueError('Expected direction backward or forward')
    S*= 1/delta_x
    return S



if __name__ == '__main__':
    T_dx = T(5,1,True)
    print("--- Testing a circulant T ---")
    print(T_dx)
    print("--- Testing a circulant, backward S ---")
    S_dx = S(5,1,'bwd',True)
    print(S_dx)
    print("--- Testing a circulant, forward S ---")
    S_dx = S(5,1,'fwd',True)
    print(S_dx)

