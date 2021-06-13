# This is the code for conjugate gradients iterative method
# William Lu
import numpy as np 
import math
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

# Setting up:
n = 8
A = np.random.standard_normal(size =(n,n))
# making it symmetric
B = (A.T +A)/2
# making it positive definite
C = B.T @ B

# Checking Positive definite
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
print(is_pos_def(C))

#finding max and min of eigenvalues of C
arr = np.linalg.eigvals(C)
list = arr.tolist()
cond_num = max(list)/min(list)

# define the right handside b1
b1 = np.random.randint(10,size =(n,1))

# First testing: xx is the cg solution, xr is the real solution, x0 is the initial guess
x0 = np.random.randint(10,size =(n,1))
xx = CG(C,x0,b1,1e-6)
xxx = np.asarray(xx) # convert tuple to array
xr = np.linalg.solve(C,b1)

# error bounds
err_bound = 2*pow((((math.sqrt(cond_num) -1))/(math.sqrt(cond_num)+1)),n)

e_k = np.subtract(xxx,xr)
e_0 = np.subtract(x0,xr)

def bounds(ek,e0,errb):
    ek_norm = np.linalg.norm(ek)
    e0_norm = np.linalg.norm(e0)
    if(ek_norm <= e0_norm*errb):
        return True
print(bounds(e_k,e_0,err_bound))
