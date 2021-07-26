from CG import create_matrix, CG, is_pos_def, bounds,cond_size,create_matrix,find_cond_num,cond_size,timing_CG
from BCG import BCG,create_NM,timing_BCG
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
    plt.plot(iteration_BCG,log_BCG_err,label ='\u2113=%1.0f'%2**expo)
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
A7 = scipy.sparse.load_npz('/Users/William/Downloads/Archive/A_n_13122_gamma_0.npz')
###########################################
# time test for matrix A1,A2,A3,A4,A5,A6. BCG. Different block size.
for expo in range(0,5):
    A1_BCG_time = timing_BCG(A1,2**expo)
    print(A1_BCG_time)

    A2_BCG_time = timing_BCG(A2,2**expo)
    print(A2_BCG_time)

    A3_BCG_time = timing_BCG(A3,2**expo)
    print(A3_BCG_time)

    A4_BCG_time = timing_BCG(A4,2**expo)
    print(A4_BCG_time)

    A5_BCG_time = timing_BCG(A5,2**expo)
    print(A5_BCG_time)

    A6_BCG_time = timing_BCG(A6,2**expo)
    print(A6_BCG_time)

#################################
# s*T(CG,1) vs. T(BCG,l) matrix A{13122,0}


iteration_timing = list(2**x for x in range(0,5)) # length of iteration
CG_tseq = []
CG_iter_tseq =[] 
for expo in range(0,5):
    A7_CG_time_1 = timing_CG(A7) 
    CG_tseq.append((2**expo)*A7_CG_time_1[0]) #  gives us the time for the algorithm of differernt blk size
    CG_iter_tseq.append((2**expo)*A7_CG_time_1[1]) #gives us the time for each iteration of matrix vector product
BCG_tseq = [] 
BCG_iter_tseq =[]
for expo in range(0,5):
    A7_BCG_time = timing_BCG(A7,2**expo)
    BCG_tseq.append(A7_BCG_time[0]) # gives us the time for the algorithm, not each iteration
    BCG_iter_tseq.append(A7_BCG_time[1]) # gives us the time for each iteration of matrix vector product

plt.plot(iteration_timing,CG_tseq,label = '\u2113*T(CG,1)')
plt.legend()
plt.plot(iteration_timing,BCG_tseq,label = 'T(BCG,\u2113)')
plt.legend()
plt.xlabel("block size")
plt.ylabel("time in seconds")
plt.title("Time for BCG and CG solving many RHS")   
plt.show()

# s*T(A,1) vs T(A,l)
plt.plot(iteration_timing,CG_iter_tseq,label = '\u2113*T(A,1)')
plt.legend()
plt.plot(iteration_timing,BCG_iter_tseq,label = 'T(A,\u2113)')
plt.legend()
plt.xlabel("block size")
plt.ylabel("time in seconds")
plt.title("Time for every iteration of BCG and CG")   
plt.show()


# s*T(CG,1) vs. T(BCG,l) matrix A{13122,1}
A8 = scipy.sparse.load_npz('/Users/William/Downloads/Archive/A_n_13122_gamma_1.npz')
#iteration_timing = list(2**x for x in range(0,5)) # length of iteration
CG_tseq_2 = []
CG_iter_tseq_2 =[] 
for expo in range(0,5):
    A8_CG_time_1 = timing_CG(A8) 
    CG_tseq_2.append((2**expo)*A8_CG_time_1[0]) #  gives us the time for the algorithm of differernt blk size
    CG_iter_tseq_2.append((2**expo)*A8_CG_time_1[1]) #gives us the time for each iteration of matrix vector product
BCG_tseq_2 = [] 
BCG_iter_tseq_2 =[]
for expo in range(0,5):
    A8_BCG_time = timing_BCG(A8,2**expo)
    BCG_tseq_2.append(A8_BCG_time[0]) # gives us the time for the algorithm, not each iteration
    BCG_iter_tseq_2.append(A8_BCG_time[1]) # gives us the time for each iteration of matrix vector product

plt.plot(iteration_timing,CG_tseq_2,label = '\u2113*T(CG,1)')
plt.legend()
plt.plot(iteration_timing,BCG_tseq_2,label = 'T(BCG,\u2113)')
plt.legend()
plt.xlabel("block size")
plt.ylabel("time in seconds")
plt.title("Time for BCG and CG solving many RHS")   
plt.show()

# s*T(A,1) vs T(A,l)
plt.plot(iteration_timing,CG_iter_tseq,label = '\u2113*T(A,1)')
plt.legend()
plt.plot(iteration_timing,BCG_iter_tseq,label = 'T(A,\u2113)')
plt.legend()
plt.xlabel("block size")
plt.ylabel("time in seconds")
plt.title("Time for every iteration of BCG and CG")   
plt.show()