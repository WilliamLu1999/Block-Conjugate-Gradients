import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import time
# M is the preconditioner.
def PBCG(Minv,A,X,B,XR,tol=1e-6,iter=None,maxiter=None):
    R = B-A.dot(X)
    Z = Minv@R
    P = Z
    k = 0 # number of iterations
    R_old  = R.T@Z # for easier future computation. numpy ndarray
    err_F_list =[] 
   
    while True: 
        A_P = A@P #  ndarray
        P_A_P_inv = np.linalg.pinv(P.T@A_P)
        Lam = P_A_P_inv@ R_old# alpha matrix
        PLam = P@Lam
        X = X + PLam
        R = R - A@(PLam)
        if iter== False:
            Xe =(XR-X).T@A@(XR-X)
            Xe_F_norm = math.sqrt(Xe.trace())
            err_F_list.append(Xe_F_norm)
    
        if np.max(np.linalg.norm(R,axis=0)) <= tol:
            break
        else:
                
            Z =Minv@R
            R_T_Z = R.T@Z
            Phi=  np.linalg.pinv(R_old)@R_T_Z

            P = Z + P@Phi
            R_old = R.T@Z #update the denominator to be the numerator's value for the next fraction
            k+=1 
            if k>=maxiter:
                break    
    if iter == True:
        return X,k
    else:
        return err_F_list

def timing_PBCG(A,P,blk):
    t_B = np.random.randint(5,size =(A.shape[0],blk)) # test CG. random column vector with size 1
    t_X = np.zeros((A.shape[0],blk))
    t_XR = scipy.sparse.linalg.spsolve(A,t_B) # real solution
    t_XR = np.reshape(t_XR,(A.shape[0],blk))
    start = time.time()
    result = PBCG(P,A,t_X,t_B,t_XR,1e-6,True,2*A.shape[0])
    end = time.time()
    iter_time = (end-start)/result[1]
    return end-start, iter_time

