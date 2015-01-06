from __future__ import division
import numpy as np
import profile
from scipy import signal , linalg
from scipy.linalg import cho_factor, cho_solve

x, y = np.meshgrid(range(-1, 2), range(-1, 2), indexing="ij")
x, y = x.flatten(), y.flatten()
AT = np.vstack((x*x, y*y, x*y, x, y, np.ones_like(x)))

def cov(f):

  nx = np.array([-1. , 0. , 1.])
  ny = nx[:,np.newaxis]
  npix = 9
  
  
  #brightness = image.flatten()
  
  xv , yv =  np.meshgrid(np.array(nx) , np.array(ny))
  xc , yc =  xv.flatten() , yv.flatten()

  C = np.ones((npix , npix))
  
  for i in range(npix):
    for j in range(npix):
      
      r2 = (xc[i] - xc[j])**2. + (yc[i] - yc[j])**2.   

      C[i,j] = np.exp(-1.*r2/(4.*f**2.))
  return C

#C = cov(1.2)
C = np.identity(9)
ATA = np.dot(AT, np.dot(np.linalg.inv(C) , AT.T))
factor = cho_factor(ATA, overwrite_a=True)
#ATA = np.dot(AT, AT.T)
#factor = cho_factor(ATA, overwrite_a=True)



def fit_3x3(im , sigma):
    #print C.shape
    imgg = np.dot(AT , np.dot(np.linalg.inv(C) , im.flatten()))
    a, b, c, d, e, f = cho_solve(factor, imgg)
    #a = a - (sigma)/11
    #b = b - (sigma)/11
    m = 1. / (4 * a * b - c*c + sigma**2./16)
    x = (c * e - 2 * b * d) * m
    y = (c * d - 2 * a * e) * m
    return x, y

def makeGaussian(size , FWHM =3 , e = 0 , center = None):

    #x = np.arange(0, size, 1, float)
    #y = x[:,np.newaxis]
    x = np.linspace(0.5, size-.5 , size)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size / 2.
    else:
        x0 = center[0]
        y0 = center[1]
    
    
    r = ((x-x0)**2. + ((y-y0)**2.)*(1. + np.abs(e))**2./(1. - np.abs(e))**2.)**.5
    factor = 1./(2.*np.pi*FWHM**2.)   
    return factor*np.exp((-1.*r**2.)/(2*FWHM**2.))


def MAD(a, axis=None):
    """Compute the median absolute deviation"""
    a = np.array(a, copy=False)
    a_median = np.median(a, axis=axis)

    #re-broadcast the output median array to subtract it
    if axis is not None:
        shape = list(a_median.shape)
        shape.append(1)
        a_median = a_median.reshape(shape)

    #calculated the median average deviation
    return np.median(np.abs(a - a_median), axis=axis)/0.6745


def find_centroid(data):

  sigma = MAD(data)
  size = data.shape[0]
  zero = size/2 + .5
  kernel = makeGaussian(7, 1.5 , 0 , np.array([3.5,3.5]))
  img = signal.convolve2d(data , kernel , mode = "same")
  xi, yi = np.unravel_index(np.argmax(img), img.shape)
    
  if (xi >= 1 and xi < img.shape[0] - 1 and yi >= 1 and yi < img.shape[1] - 1):
      ox, oy = fit_3x3(img[xi-1:xi+2, yi-1:yi+2] , sigma)
  else:
      ox , oy = 0. , 0.
  return ox , oy

 

if __name__ == "__main__":
    print 'c3 main'
