import numpy as np
import scipy.sparse.linalg
import scipy.sparse.csr_matrix
import math
A = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/A_n_882_gamma_0.npz')
M =scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/P_n_882_gamma_0.npz')
#z = np.linalg.pinv(A)
#print(A.get_shape())
#print(A)
#C = scipy.sparse.linalg.inv(M)
#print(C)
#print(M.get_shape())
#print(C*A)
#print(type(A.tocsc()))
b = np.random.randint(10,size =(882,1))
x = np.zeros((882,1))
#print(type(A.multiply(b)))
xr=scipy.sparse.linalg.cg(A,b,M=M) # real solution

def PCG(A,x,b,M,xr,tol=1e-6,iter=None):
    A1= A.tocsc()
    r = b-A1.dot(x)
    z = scipy.sparse.linalg.inv(M).multiply(r) # z is in coocoo sparse format
    p = r
    k = 0 # number of iterations
    r_old  = r.T.dot(z)
    #print(r_old)
    err_list =[] 
    
    while True: 
        A_p = A1.dot(p) # ~ n^2 FLOPs
        alpha = r_old/np.dot(p.T,A_p) # ~ n FLOPs
        x = x + alpha*p # ~ n FLOPs
        r = r - alpha*A_p
        x_ii = x - xr
        print(x_ii)
        #print(len(x_ii))
        # finding the energy norm of x_ii
        A_xii = A1.dot(x_ii)
        #print(len(A_xii))
        energy_norm_xi = math.sqrt(np.dot(x_ii.T,A_xii))
        #print(len(np.dot(x_ii.T,A_xii)))
        err_list.append(energy_norm_xi)
        if np.linalg.norm(r) <= tol:
            break
        else:
            z =scipy.sparse.inv(M).multiply(r)
            beta = np.dot(r.T,z)/r_old # ~ n FLOPs
            p = z + beta*p
            r_old = np.dot(r.T,z) #update the denominator to be the numerator's value for the next fraction
            k += 1     
    if iter == True:
        return x  
    else:
        return err_list


sol =PCG(A,x,b,M,xr,1e-6,True)
print(sol)

