"""Shifts the model to the center of stars"""


from __future__ import division
from numpy import arange, array, linspace, ones, zeros
from scipy.linalg import solve_banded
import interp as _spline
import c3

def cal_coefs(a, b, y, c=None, alpha=0, beta=0):
    '''
    Return spline coefficients 

    Parameters
    ----------
    a : float
        lower bound of the grid.
    b : float
        upper bound of the grid.
    y : ndarray
        actual function value at grid points.
    c : (y.shape[0] + 2, ) ndarray, optional
        ndarry to be written
    alpha : float
        Second-order derivative at a. Default is 0.
    beta : float
        Second-order derivative at b. Default is 0.

    Returns
    -------
    out : ndarray
        Array of coefficients.
    '''
    n = y.shape[0] - 1
    h = (b - a)/n

    if c is None:
        c = zeros((n + 3, ))
        ifreturn = True
    else:
        assert(c.shape[0] == n + 3)
        ifreturn = False

    c[1] = 1/6 * (y[0] - (alpha * h**2)/6)
    c[n + 1] = 1/6 * (y[n] - (beta * h**2)/6)

    # ab matrix here is just compressed banded matrix
    ab = ones((3, n - 1))
    ab[0, 0] = 0
    ab[1, :] = 4
    ab[-1, -1] = 0

    B = y[1:-1].copy()
    B[0] -= c[1]
    B[-1] -=  c[n + 1]

    c[2:-2] = solve_banded((1, 1), ab, B)

    c[0] = alpha * h**2/6 + 2 * c[1] - c[2]
    c[-1] = beta * h**2/6 + 2 * c[-2] - c[-3]

    if ifreturn:
        return(c)

# aliases
interpolate = _spline.interpolate
interpolate_2d = _spline.interpolate_2d
cen = c3.find_centroid
def centerer(y):
  
  """
  input = image
  output = centered image using cubic-spline INTERPOLATION
  """

  a1 , a2 = 0.5 , 0.5
  b1 , b2 = y.shape[0] - .5 , y.shape[1] - .5
  n1 , n2 = y.shape[0] - 1 , y.shape[1] - 1     #image.shape[0]([1]) = 
                                                      #number of input grid
                                                      #points along the x(y) axis
  h1 , h2 = (b1 - a1)/n1, (b2 - a2)/n2
  grid_x = arange(n1 + 1)* h1 + a1
  grid_z = arange(n2 + 1)* h2 + a2
  
  alpha =  0.        #second derivative of spline at the start point
  beta  =  0.        #second derivative of spline at the end point

  c_tmp = zeros((n1 + 3 , n2 + 1))
  cal_coefs(a1 , b1 , y , c_tmp)
  
  c = zeros((n1+3 , n2+3))           #initializing the place holder for spline coefficients
  cal_coefs(a2 , b2 , c_tmp.T , c.T)
  
  
  ox , oz = cen(y)
  
  gridhat_x = grid_x - ox
  gridhat_z = grid_z - oz
  #print grid_x
  #print gridhat_x

  yhat = zeros((n1+1 , n2+1))
  
  for i, x in enumerate(gridhat_x):
    for j, z in enumerate(gridhat_z):
      yhat[i, j] = interpolate_2d(x, z, a1, b1, a2, b2, c)

  return yhat
