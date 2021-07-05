from CG import create_matrix, CG, is_pos_def, bounds,cond_size,create_matrix,find_cond_num,cond_size
from BCG import BCG,create_NM
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse.linalg

# Setup
A = create_matrix(10)
X = create_NM(10,5)
B = create_NM(10,5)
# Testing
XK = BCG(A,X,B,1e-6,len(X),True)
XR =np.linalg.solve(A,B)
if(XK.all()==XR.all()):
    print('Your BCG converges to the right solution.')

##############################################################
# block size vs convergence
# decide the matrix size. create matrix W
n  = 80
W = create_matrix(n)

# Compare block size 1 in BCG to CG
# First testing: x0 is the initial guess, b1 is a block size of 1 vector
x0 = np.random.randint(10,size =(n,1))
b1 = np.random.randint(10,size =(n,1))
xxr = np.linalg.solve(W,b1)
cg_err = CG(W,x0,b1,xxr,1e-6,False)
log_energy_cg= [math.log10(j) for j in cg_err]
iteration_cg = list(range(0,len(cg_err)))
plt.plot(iteration_cg,log_energy_cg,label ='cg line')

bcg_err = BCG(W,x0,b1,1e-6,2*n,False)
log_energy_bcg= [math.log10(p) for p in bcg_err]
iteration_bcg = list(range(0,len(bcg_err)))
plt.plot(iteration_bcg,log_energy_bcg,label ='bcg line')
plt.legend()
plt.xlabel("iteration")
plt.ylabel("error log10 base")
plt.title("BCG versus CG of block size 1")   
plt.show()

###############################################################
# Block size of 2,4,8,16
for expo in range(1,5):
    X0 = np.random.randint(10,size=(n,2**expo))
    B1 = np.random.randint(10,size=(n,2**expo))
    BCG_err = BCG(W,X0,B1,1e-6,2*n,False)
    log_frob_BCG = [math.log10(q) for q in BCG_err]
    iteration_BCG = list(range(0,len(BCG_err)))
    plt.plot(iteration_BCG,log_frob_BCG,label =2**expo)
    plt.legend()
plt.xlabel("iteration")
plt.ylabel("error log10 base")
plt.title("BCG: different block size")   
plt.show()

#################################################################
#graph error function, assume kappa being 10000
u = np.linspace(-2,5000,100)
y_1 = 2*(99/100)**u
y_2 = 4*(0.99/1.01)**(2*u)
plt.plot(u,y_1,label ="CG")
plt.legend()
plt.plot(u,y_2,label='PCG')
plt.legend()
plt.show()