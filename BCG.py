from CG import create_matrix
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse.linalg
def BCG(A,X,B,Xr,tol =1e-6,maxiter = None,iter=None):
    R = B - A@X #Residual
    P = R #Search direction
    k = 0 #iteration
    R_old = R.T@R #denominator(for easier calculation later)
    err_F_list= []
    
    while True:
    
        A_P =A@P
        P_A_P_inv = np.linalg.pinv(P.T@A_P)
        # Lam is the step size matrix
        Lam = P_A_P_inv @ R_old
        PLam = P@Lam
        X = X + PLam
        R = R - A@(PLam)

        # Find the error and the Frobenius norm
        if iter== False:
            Xe =(Xr-X).T@A@(Xr-X)
            Xe_F_norm = math.sqrt(Xe.trace())
            err_F_list.append(Xe_F_norm)
        if np.linalg.norm(R) <= tol:
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

