import numpy as np
from stuff import stuff
import shift

data = np.loadtxt("varpsf.txt")
h = stuff(data, 2 , min_iter=1, max_iter=20, check_iter=1, tol=1.e-12)
