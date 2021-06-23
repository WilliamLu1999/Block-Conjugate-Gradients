from CG import create_matrix
from BCG import BCG,create_NM,find_Frobenius
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse.linalg

# Setup
A = create_matrix(10)
X = create_NM(10,5)
B = create_NM(10,5)
# Testing
XK = BCG(A,X,B,1e-6,len(X))
XR =np.linalg.solve(A,B)
if(XK.all()==XR.all()):
    print('Your BCG converges to the right solution.')

##############################################################
# block size vs convergence
# decide the matrix size
n  = np.random.randint(2,100)
# create matrix W
W = create_matrix(n)
F_list= []
for j in range(1,10):
    blk_size = np.random.randint(2,800) # block size
    BB = create_NM(n,blk_size) # Righ hand side matrix B
    XX = create_NM(n,blk_size) # Guessed initial solution X
    F_list = find_Frobenius(A,X,B,1e-6)
    log_Frobenius =[math.log10(f) for f in F_list]
    iteration4 = list(range(0,len(F_list)))
    plt.plot(iteration4,log_Frobenius,label = blk_size)
    plt.legend()
#print(len(F_list))
plt.xlabel("iteration")
plt.ylabel("error log10 base")
plt.title("Block size versus convergence speed")    
plt.show()
plt.savefig('blockCS.pgf')

