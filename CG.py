# This is the code for conjugate gradients iterative method
# William Lu
import numpy as np 
def CG(A, x, b, tolerance=1e-6):
    # A is the matrix, x is the solution, b is the right handside
    # tol is the boundary, n is the size
    r = b-A.dot(x)
    # if r is very small, return x
    ## if r<= 0.001:
        return x
    p = r
    k = 0 # number of iterations
    while True: # Each iteration: ~ n^2 FLOPs
        A_p = A.dot(p) # ~ n^2 FLOPs
        alpha = np.dot(p,r)/np.dot(p,A_p) # ~ n FLOPs
        x = x + alpha*p # ~ n FLOPs
        r = b - A.dot(x) # ~ n^2 FLOPs
        if np.linalg.norm(r) <= tolerance:
            k+=1
            break
        else:
            beta = -np.dot(r,A_p)/np.dot(p,A_p) # ~ n FLOPs
            p = r + beta*p
            k += 1
    return x        

B =np.array([[1,2],[3,5]])
print(type(B))
y = np.array([[4],[7]])
c = np.array([[3],[11]])
a = CG(B,y,c,)
print(a)
