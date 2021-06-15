# This is the code for conjugate gradients iterative method
# William Lu
import numpy as np 
import math
import matplotlib.pyplot as plt
def CG(A, x, b, tolerance=1e-6):
    # A is the matrix, x is the solution, b is the right handside
    # tol is the boundary, n is the size
    r = b-A.dot(x)
    # if r is very small, return x
    ## if r<= 0.001:
    p = r
    k = 0 # number of iterations
   
    while True: # Each iteration: ~ n^2 FLOPs
        A_p = A.dot(p) # ~ n^2 FLOPs
        alpha = np.dot(p.T,r)/np.dot(p.T,A_p) # ~ n FLOPs
        x = x + alpha*p # ~ n FLOPs
        r = b - A.dot(x) # ~ n^2 FLOPs
        if np.linalg.norm(r) <= tolerance:
            k+=1
            break
        else:
            beta = -np.dot(r.T,A_p)/np.dot(p.T,A_p) # ~ n FLOPs
            p = r + beta*p
            k += 1
    return x        

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
    list = arr.tolist()
    cond_num = max(list)/min(list)
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

# xr is the real solution, xi is the solution of every iteration. 
# we want to calculate energy norm of each iteration and plot the graph
def find_error(A, xi, b, tolerance=1e-6):
    r = b-A.dot(xi)
    p = r
    k = 0 # number of iterations
    err_list =[]
    while True: # Each iteration: ~ n^2 FLOPs
        A_p = A.dot(p) # ~ n^2 FLOPs
        alpha = np.dot(p.T,r)/np.dot(p.T,A_p)

      
        
        xr = np.linalg.solve(A,b)
        # x_ii is the error at i^th iteration
        x_ii = xi - xr
        # finding the energy form of x_ii
        A_xii = A.dot(x_ii)
        energy_norm_xi = math.sqrt(np.dot(x_ii.T,A_xii))
        err_list.append(energy_norm_xi)


        xi = xi + alpha*p # ~ n FLOPs

        r = b - A.dot(xi) # ~ n^2 FLOPs
        if np.linalg.norm(r) <= tolerance:
            k+=1
            break
        else:
            beta = -np.dot(r.T,A_p)/np.dot(p.T,A_p) # ~ n FLOPs
            p = r + beta*p
            k += 1

    return err_list
    

   





