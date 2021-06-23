from CG import create_matrix
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse.linalg
def BCG(A,X,B,tol =1e-6,maxiter = None):
    R = B - A@X #Residual
    P = R #Search direction
    k = 0 #iteration
    R_old = R.T@R #denominator(for easier calculation later)
    while True:
    
        A_P =A@P
        P_A_P_inv = np.linalg.pinv(P.T@A_P)
        # Lam is the step size matrix
        Lam = P_A_P_inv @ R_old
        X = X + P@Lam
        R = R - A@(P@Lam)
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
            
    return X  

def create_NM(n,m): #n is the size of the matrix A; m is the number of columns of intial guess
    X = np.random.uniform(0,10,(n,m))
    return X

def find_Frobenius(A,X,B,tol =1e-6):
    R = B - A@X #Residual
    P = R #Search direction
    R_old = R.T@R #denominator(for easier calculation later)
    Xr = np.linalg.solve(A,B)
    err_F_list= []
    counter = 0
    while R.all()!= 0:
        A_P =A@P
        P_A_P_inv = np.linalg.pinv(P.T@A_P)
        # Lam is the step size matrix
        Lam = P_A_P_inv @ R_old
        X = X + P@Lam
        R = R - A@(P@Lam)

        # Find the error and the Frobenius norm
        Xe = X - Xr
        Xe_F_norm = np.linalg.norm(Xe)
        err_F_list.append(Xe_F_norm)

        if np.linalg.norm(R)<=tol:
            break
        else:
            R_T_R = R.T@R
            Phi=  np.linalg.pinv(R_old)@R_T_R
            P = R+ P@Phi
            R_old = R.T@R #update the denominator to be the numerator's value for the next fraction
            counter+=1
    return err_F_list




