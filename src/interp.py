
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


def _interpolate_2d(x, y, a1, b1, a2, b2, c)

  n1 = c.shape[0] - 3
  n2 = c.shape[1] - 3
  h1 = (b1 - a1)/n1
  h2 = (b2 - a2)/n2
  l1 = np.int((x - a1)//h1) + 1
  l2 = np.int((y - a2)//h2) + 1
  m1 = np.int(np.min(l1 + 3, n1 + 3))
  m2 = np.int(np.min(l2 + 3, n2 + 3))
  
  s = 0

 for i1 in xrange(l1, m1 + 1):
    u_x = u(x, i1, a1, h1)
    for i2 in xrange(l2, m2 + 1):
      u_y = u(y, i2, a2, h2)
      s += c[i1 - 1, i2 - 1] * u_x * u_y

  return s
