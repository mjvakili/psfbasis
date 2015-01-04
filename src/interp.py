
import numpy as np


def PI(t):
  
  abt = np.abs(t)
  if (abt <= 1):
    return 4 - 6 * abt**2 + 3 * abt **3
  elif (abt <= 2):
    return (2 - abt)**3
  else:
    return 0
    
    
def u(x, k, a, h):
    
    """ x : float, k: integer,
    a: float, h : float """ 
  
    return PI((x - a)/h - (k - 2))
  
def _interpolate(x, a, b, c):
    """ x : float,
        a : float, lower bound
        b: float, upper bound
        c: array"""
    
    n = c.shape[0] - 3
    h = (b - a)/n
    l = np.int((x - a)//h) + 1
    m = np.int(np.min(l + 3, n + 3))
    s = 0
    for k in xrange(l, m + 1):
        s += c[k - 1] * u(x, k, a, h)

    return s
    
def interpolate(x, a, b, c):
    '''
    Return interpolated function value at x
    
    Parameters
    ----------
    x : float
        The value where the function will be approximated at
    a : double
        Lower bound of the grid
    b : double
        Upper bound of the grid
    c : ndarray
        Coefficients of spline
    
    Returns
    -------
    out : float
        Approximated function value at x
    '''
    
    return _interpolate(x, a, b, c)
