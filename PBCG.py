import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse.linalg
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
        X = X + P@Lam
        R = R - A@(P@Lam)
        
        Xe =(XR-X).T@A@(XR-X)
        Xe_F_norm = math.sqrt(Xe.trace())
        err_F_list.append(Xe_F_norm)
        if np.linalg.norm(R) <= tol:
            break
        else:
            Z =Minv@R
            R_T_Z = R.T@Z
            Phi=  np.linalg.pinv(R_old)@R_T_Z

            P = Z + P@Phi
            R_old = R.T@Z #update the denominator to be the numerator's value for the next fraction
            k += 1 
            if k>=maxiter:
                break    
    if iter == True:
        return X,k
    else:
        return err_F_list
