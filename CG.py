# This is the code for conjugate gradients iterative method
# William Lu
import numpy as np 
import math
import matplotlib.pyplot as plt
import scipy as sp
import inspect
def CG(A, x, b, xr, tolerance=1e-6,iter=None):
    # A is the matrix, x is the solution, b is the right handside
    # tol is the boundary, n is the size
    r = b-A.dot(x)
    # if r is very small, return x
    ## if r<= 0.001:
    p = r

    k = 0 # number of iterations
    r_old  = np.dot(r.T,r)
    err_list =[]
    while True: # Each iteration: ~ n^2 FLOPs

        A_p = A.dot(p) # ~ n^2 FLOPs
        alpha = r_old/np.dot(p.T,A_p) # ~ n FLOPs
        x = x + alpha*p # ~ n FLOPs
        
        r = r - alpha*A_p
        if iter == False:
            x_ii = x - xr
            # finding the energy norm of x_ii
            A_xii = A.dot(x_ii)
            energy_norm_xi = math.sqrt(np.dot(x_ii.T,A_xii))
            err_list.append(energy_norm_xi)
        if np.linalg.norm(r) <= tolerance:
            break
        else:
            beta = np.dot(r.T,r)/r_old # ~ n FLOPs
            p = r + beta*p
            r_old = np.dot(r.T,r) #update the denominator to be the numerator's value for the next fraction
            k += 1     
    if iter == True:
        return x,k
    else:
        return err_list

def create_matrix(n):
    A = np.random.standard_normal(size =(n,n))
    # making it symmetric
    B = (A.T +A)/2
    # making it positive definite
    C = B.T @ B
    return C

# C is positive definite matrix
def find_cond_num(C):
    arr = np.linalg.eigvals(C)
    cond_num = np.max(arr)/np.min(arr)
    return cond_num

# Checking Positive definite
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# Testing if the bounds in the book works
def bounds(ek,e0,errb):
    ek_norm = np.linalg.norm(ek)
    e0_norm = np.linalg.norm(e0)
    if(ek_norm <= e0_norm*errb):
        return True

# graphing the relationship between matrix size and its condition number
# https://towardsdatascience.com/creating-custom-plotting-functions-with-matplotlib-1f4b8eba6aa1
def cond_size(x,y, ax=None, **plt_kwargs):
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, **plt_kwargs) 
    return(ax)

    


