from CG import create_matrix, CG, is_pos_def, bounds,cond_size,create_matrix,find_cond_num,cond_size
from BCG import BCG,create_NM
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import time

# Setup
A = create_matrix(10)
X = create_NM(10,5)
B = create_NM(10,5)
# Testing
XR =np.linalg.solve(A,B)
XK = BCG(A,X,B,XR,1e-6,len(X),True)


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

bcg_err = BCG(W,x0,b1,xxr,1e-6,2*n,False)
log_energy_bcg= [math.log10(p) for p in bcg_err]
iteration_bcg = list(range(0,len(bcg_err)))
plt.plot(iteration_bcg,log_energy_bcg,label ='bcg line')
plt.legend()
plt.xlabel("iteration")
plt.ylabel("error log10 base")
plt.title("BCG versus CG of block size 1")   
plt.show()

###############################################################
# Block size of 1,2,4,8,16
for expo in range(0,5):
    X0 = np.random.randint(10,size=(n,2**expo))
    B1 = np.random.randint(10,size=(n,2**expo))
    xR = np.linalg.solve(W,B1)
    BCG_err = BCG(W,X0,B1,xR,1e-6,2*n,False)
    log_frob_BCG = [math.log10(q) for q in BCG_err] # getting log norm
    relative_log_frob_BCG = [h/log_frob_BCG[0] for h in log_frob_BCG] # getting relative error
    log_BCG_err = [math.log10(h/BCG_err[0]) for h in BCG_err]
    iteration_BCG = list(range(0,len(BCG_err))) # length of iteration
    plt.plot(iteration_BCG,log_BCG_err,label ='b=%1.0f'%2**expo)
    plt.legend()
    #start = time.time()
    BCG(W,X0,B1,xR,1e-6,2*n,True)
    #end = time.time()
    #print(end - start)
    #print(BCG(W,X0,B1,xR,1e-6,2*n,True))
plt.xlabel("iteration")
plt.ylabel("Relative error log10 base")
plt.title("BCG: different block size")   
plt.show()

#################################################################
# for making the table. time.

A1 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/A_n_882_gamma_0.npz')
A2 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/A_n_882_gamma_1.npz')
A3 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/A_n_882_gamma_1000.npz')
A4 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/A_n_3362_gamma_0.npz')
A5 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/A_n_3362_gamma_1.npz')
A6 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/A_n_3362_gamma_1000.npz')

'''
for expo in range(0,5):
    BB = np.random.randint(5,size =(A1.shape[0],2**expo))
    XX = np.zeros((A1.shape[0],2**expo))
    XXR1 = scipy.sparse.linalg.spsolve(A1,BB) # real solution
    XXR1 = np.reshape(XXR1,(A1.shape[0],2**expo))
    start = time.time()
    BCG(A1,XX,BB,XXR1,1e-6,2*A1.shape[0],True)
    end = time.time()
    print(end - start)
    print(BCG(A1,XX,BB,XXR1,1e-6,2*A1.shape[0],True))
   ''' 
for expo in range(0,5):
    BB2 = np.random.randint(5,size =(A2.shape[0],2**expo))
    XX2 = np.zeros((A2.shape[0],2**expo))
    XXR2 = scipy.sparse.linalg.spsolve(A2,BB2) # real solution
    XXR2 = np.reshape(XXR2,(A2.shape[0],2**expo))
    start = time.time()
    BCG(A2,XX2,BB2,XXR2,1e-6,2*A2.shape[0],True)
    end = time.time()
    #print(end - start)
    #print(BCG(A2,XX2,BB2,XXR2,1e-6,2*A2.shape[0],True))

for expo in range(0,5):
    BB3 = np.random.randint(5,size =(A3.shape[0],2**expo))
    XX3 = np.zeros((A3.shape[0],2**expo))
    XXR3 = scipy.sparse.linalg.spsolve(A3,BB3) # real solution
    XXR3 = np.reshape(XXR3,(A2.shape[0],2**expo))
    start = time.time()
    BCG(A3,XX3,BB3,XXR3,1e-6,2*A3.shape[0],True)
    end = time.time()
    #print(end - start)
    #print(BCG(A3,XX3,BB3,XXR3,1e-6,2*A3.shape[0],True))
    
for expo in range(0,5):
    BB4 = np.random.randint(5,size =(A4.shape[0],2**expo))
    XX4 = np.zeros((A4.shape[0],2**expo))
    XXR4 = scipy.sparse.linalg.spsolve(A4,BB4) # real solution
    XXR4 = np.reshape(XXR4,(A4.shape[0],2**expo))
    start = time.time()
    BCG(A4,XX4,BB4,XXR4,1e-6,2*A4.shape[0],True)
    end = time.time()
    #print(end - start)
    #print(BCG(A4,XX4,BB4,XXR4,1e-6,2*A4.shape[0],True))
for expo in range(0,5):
    BB5 = np.random.randint(5,size =(A5.shape[0],2**expo))
    XX5 = np.zeros((A5.shape[0],2**expo))
    XXR5 = scipy.sparse.linalg.spsolve(A5,BB5) # real solution
    XXR5 = np.reshape(XXR5,(A5.shape[0],2**expo))
    start = time.time()
    BCG(A5,XX5,BB5,XXR5,1e-6,2*A5.shape[0],True)
    end = time.time()
    #print(end - start)
    #print(BCG(A5,XX5,BB5,XXR5,1e-6,2*A5.shape[0],True))
for expo in range(0,5):
    BB6 = np.random.randint(5,size =(A6.shape[0],2**expo))
    XX6 = np.zeros((A6.shape[0],2**expo))
    XXR6 = scipy.sparse.linalg.spsolve(A6,BB6) # real solution
    XXR6 = np.reshape(XXR6,(A6.shape[0],2**expo))
    start = time.time()
    BCG(A6,XX6,BB6,XXR6,1e-6,2*A6.shape[0],True)
    end = time.time()
    #print(end - start)
    #print(BCG(A6,XX6,BB6,XXR6,1e-6,2*A6.shape[0],True))


#################################
# l*T(CG,1) vs. T(BCG,l)
t_BB2 = np.random.randint(5,size =(A2.shape[0],1)) # test CG. random column vector with size 1
t_XX2 = np.zeros((A2.shape[0],1))
t_XXR2 = scipy.sparse.linalg.spsolve(A2,cg_BB2) # real solution
t_XXR2 = np.reshape(cg_XXR2,(A1.shape[0],1))
start = time.time()
CG(A2,t_XX2,t_BB2,t_XXR2,1e-6,True)
end = time.time()
print(end - start) #0.012698173522949219
start2 = time.time()
BCG(A1,cg_XX2,cg_BB2,cg_XXR2,1e-6,2*A2.shape[0],True)
end2 = time.time()
print(end2 - start2) #


'''
plt.plot(iteration_BCG,log_BCG_err,label ='b=%1.0f'%2**expo)
plt.xlabel("block size")
plt.ylabel("time in seconds")
plt.title("BCG versus")   
plt.show()'''

J =[[1,2],
[3,5]]
start3 = time.time()
j1 = np.linalg.inv(J)
end3 = time.time()
print(end3-start3)

start4 = time.time()
j2 = np.linalg.pinv(J)
end4 = time.time()
print(end3-start3)