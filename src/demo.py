import numpy as np
from stuff import stuff
import shift
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':12})
rc('text', usetex=True)
from matplotlib import cm
import numpy as np

data = np.loadtxt("varpsf.txt")
Q = 6
h = stuff(data, 2 , min_iter=1, max_iter=20, check_iter=1, tol=1.e-12)
for j in range(maxiter):
  
  img = np.loadtxt("G_6(%d).txt"(%range(maxiter)[j]+1)).reshape(Q,15,15)
  fig , axarr = plt.subplots(1,6) 


  for i in range(Q):

    axarr[i].imshow(img[i] , interpolation = "None" , origin = "lower" , cmap = cm.Greys_r)
    axarr[i].set_xlim([-.5,14.5])
    axarr[i].set_ylim([-.5,14.5])
    axarr[i].set_xticks(())
    axarr[i].set_yticks(())
  plt.tight_layout()
  fig.set_size_inches(28,4)

  plt.savefig("G_6_(%d).png"(%range(maxiter)[j]+1)) , dpi=200)
