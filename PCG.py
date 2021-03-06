import numpy as np
import scipy.sparse.linalg
import math
import time
def PCG(A,x,b,Minv,xr,tol=1e-6,iter=None):
    r = b-A.dot(x)
    z = Minv@r # z is in coocoo sparse format
    
    p = z
    k = 0 # number of iterations
    r_old  = np.dot(r.T,z) # for easier future computation. numpy ndarray
    err_list =[] 
   
    while True: 
        A_p = A.dot(p) #  ndarray
        alpha = r_old/np.dot(p.T,A_p) # denominator is ndarray
        x = x + alpha*p
        r = r - alpha*A_p
        if iter == False:
            x_ii = x - xr
            # finding the energy norm of x_ii
            A_xii = A.dot(x_ii)
            energy_norm_xi = math.sqrt(np.dot(x_ii.T,A_xii))
            #print(len(np.dot(x_ii.T,A_xii)))
            err_list.append(energy_norm_xi)
        if np.linalg.norm(r) <= tol:
            break
        else:
            z =Minv@r
            beta = np.dot(r.T,z)/r_old # ~ n FLOPs
            p = z + beta*p
            r_old = np.dot(r.T,z) #update the denominator to be the numerator's value for the next fraction
            k += 1     
    if iter == True:
        return x,k
    else:
        return err_list


def timing_PCG(A,P):
    t_b = np.random.randint(5,size =(A.shape[0],1)) # test CG. random column vector with size 1
    t_x = np.zeros((A.shape[0],1))
    t_xr = scipy.sparse.linalg.spsolve(A,t_b) # real solution
    t_xr = np.reshape(t_xr,(A.shape[0],1))
    start = time.time()
    result =PCG(A,t_x,t_b,P,t_xr,1e-6,True)
    end = time.time()
    iter_time = (end-start)/result[1]
    return end-start, iter_time