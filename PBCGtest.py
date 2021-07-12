from PBCG import PBCG 
from PCG import PCG
from BCG import BCG
from CG import create_matrix
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse.linalg 
from scipy.sparse import identity

A1 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/A_n_882_gamma_0.npz')
Minv1 =scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/P_n_882_gamma_0.npz')

A4 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/A_n_3362_gamma_0.npz')
Minv4 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/P_n_3362_gamma_0.npz')

b = np.random.randint(5,size =(A1.shape[0],1))
x = np.zeros((A1.shape[0],1))
xr = scipy.sparse.linalg.spsolve(A1,b) # real solution
xr = np.reshape(xr,(A1.shape[0],1)) # make the dimension correct

# Test 1: PBCG with block size of 1. It should be the same as PCG.
result_PBCG = PBCG(Minv1,A1,x,b,xr,1e-6,False,2*A1.shape[0])
result_PCG = PCG(A1,x,b,Minv1,xr,1e-6,False)
#print(result_PBCG)
#print(result_PCG)

#graph:
plt.xlabel("iteration")
plt.ylabel("error theoretical log10 base")
plt.title("PBCG bs PCG of block size one")

log_pbcg = [math.log10(i) for i in result_PBCG]
iteration_pbcg = list(range(0,len(result_PBCG)))

plt.plot(iteration_pbcg,log_pbcg,label =('PBCG,n=882, \u03B3 =0'))
plt.legend()
log_pcg = [math.log10(i) for i in result_PCG]
iteration_pcg = list(range(0,len(result_PCG)))
plt.plot(iteration_pcg,log_pcg,label =('PCG,n=882, \u03B3 =0'))
plt.legend()
plt.show()
# Test 2: Identity matrix will give the exact x as the rigth hand side.
I_1 = scipy.sparse.identity(A1.shape[0],format ='csr')
b_2 = np.random.randint(5,size =(I_1.shape[0],2))
x_2 = np.zeros((I_1.shape[0],2))
XR_2 = scipy.sparse.linalg.spsolve(A1,b_2) # real solution
XR_2 = np.reshape(XR_2,(A1.shape[0],2)) # make the dimension correct

result_PBCG_2 = PBCG(I_1,A1,x_2,b_2,XR_2,1e-6,False,2*I_1.shape[0])
result_BCG_2 = BCG(A1,x_2,b_2,XR_2,1e-6,2*A1.shape[0],False)

plt.xlabel("iteration")
plt.ylabel("error theoretical log10 base")
plt.title("PBCG with identity preconditioner and BCG")
log_pbcg_2 = [math.log10(i) for i in result_PBCG_2]
iteration_pbcg_2 = list(range(0,len(result_PBCG_2)))
plt.plot(iteration_pbcg_2,log_pbcg_2,label =('PBCG,n=882, \u03B3 =0'))
plt.legend()
log_bcg_2 = [math.log10(i) for i in result_BCG_2]
iteration_bcg_2 = list(range(0,len(result_BCG_2)))
plt.plot(iteration_bcg_2,log_bcg_2,label =('BCG,n=882, \u03B3 =0'))
plt.legend()
plt.show()

#print(type(I_1))
#print(b_2)
#print(XR_2)
#print(result_PBCG_2)

# Test 3: Increase block sizes; matrix size: 882

for expo in range(0,5):
    X0 = np.random.randint(10,size=(A1.shape[0],2**expo))
    B1 = np.random.randint(10,size=(A1.shape[0],2**expo))
    XRs = scipy.sparse.linalg.spsolve(A1,B1)
    XRs= np.reshape(XRs,(A1.shape[0],2**expo))
    PBCG_err = PBCG(Minv1,A1,X0,B1,XRs,1e-6,False,2*A1.shape[0])
    log_frob_PBCG = [math.log10(q) for q in PBCG_err]
    relative_log_frob_PBCG = [h/log_frob_PBCG[0] for h in log_frob_PBCG]
    iteration_PBCG = list(range(0,len(relative_log_frob_PBCG)))
    plt.plot(iteration_PBCG,relative_log_frob_PBCG,label =('PBCG, n=882, \u03B3 =0, b=',2**expo))
    plt.legend()
plt.xlabel("iteration")
plt.ylabel("error log10 base")
plt.title("PBCG: different block size")   
plt.show()


# Test 4: Increase block sizes; matrix size: 3362
for expo in range(0,5):
    X00 = np.random.randint(10,size=(A4.shape[0],2**expo))# random solution
    B11 = np.random.randint(10,size=(A4.shape[0],2**expo))# random right hand side
    XRss = scipy.sparse.linalg.spsolve(A4,B11) # true solution
    XRss= np.reshape(XRss,(A4.shape[0],2**expo))
    PBCG_err_4 = PBCG(Minv4,A4,X00,B11,XRss,1e-6,False,2*A4.shape[0])
    log_frob_PBCG_4 = [math.log10(q) for q in PBCG_err_4]
    relative_log_PBCG_4 = [h/log_frob_PBCG_4[0] for h in log_frob_PBCG_4]
    iteration_PBCG_4 = list(range(0,len(relative_log_PBCG_4)))
    plt.plot(iteration_PBCG_4,relative_log_PBCG_4,label =('PBCG, n=3362, \u03B3 =0, b=',2**expo))
    plt.legend()
plt.xlabel("iteration")
plt.ylabel("error log10 base")
plt.title("PBCG: different block size")  

plt.show()

