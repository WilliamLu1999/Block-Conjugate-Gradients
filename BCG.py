from CG import create_matrix
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import time
def BCG(A,X,B,Xr,tol =1e-6,maxiter = None,iter=None):
    R = B - A@X #Residual
    P = R #Search direction
    k = 0 #iteration
    R_old = R.T@R #denominator(for easier calculation later)
    err_F_list= []
    
    while True:
    
        A_P = A@P
        P_A_P_inv = np.linalg.pinv(P.T@A_P)
        # Lam is the step size matrix
        Lam = P_A_P_inv @ R_old
        PLam = P@Lam
        X = X + PLam
        R = R - A_P@Lam

        # Find the error and the Frobenius norm
        if iter== False:
            Xe =(Xr-X).T@A@(Xr-X)
            Xe_F_norm = math.sqrt(Xe.trace())
            err_F_list.append(Xe_F_norm)
        if np.max(np.linalg.norm(R,axis=0)) <= tol:
            break
        else:
            R_T_R = R.T@R
            Phi=  np.linalg.pinv(R_old)@R_T_R
            P = R+ P@Phi
            R_old = R.T@R #update the denominator to be the numerator's value for the next fraction
            k+=1
            if k>=maxiter:
                break
    if iter==True:        
        return X,k  
    else:
        return err_F_list

def create_NM(n,m): #n is the size of the matrix A; m is the number of columns of intial guess
    X = np.random.uniform(0,10,(n,m))
    return X

def timing_BCG(A,blk): # blk is the block size
    t_B = np.random.randint(5,size =(A.shape[0],blk)) # test CG. random column vector with size 1
    t_X = np.zeros((A.shape[0],blk))
    t_XR = scipy.sparse.linalg.spsolve(A,t_B) # real solution
    t_XR = np.reshape(t_XR,(A.shape[0],blk))
    start = time.time()
    result = BCG(A,t_X,t_B,t_XR,1e-6,2*A.shape[0],True)
    end = time.time()
    iter_time = (end-start)/result[1]
    return end-start, iter_time
