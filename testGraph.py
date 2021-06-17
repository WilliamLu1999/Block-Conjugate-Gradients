from CG import CG, is_pos_def, bounds,cond_size,create_matrix,find_cond_num,cond_size,find_error
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline
# Setting up:
n = 80
C = create_matrix(n)

#condition num of C
kappa = find_cond_num(C)

# define the right handside b1
b1 = np.random.randint(10,size =(n,1))

# First testing: xx is the cg solution, xr is the real solution, x0 is the initial guess
x0 = np.random.randint(10,size =(n,1))
xx = CG(C,x0,b1,1e-6)
xxx = np.asarray(xx) # convert tuple to array
xr = np.linalg.solve(C,b1)


# Theory error plotting
C_x_0 = C.dot(x0) 
energy_norm_0 = math.sqrt(np.dot(x0.T,C_x_0)) #get A norm of the error of initial guess
iteration_k = list(range(0,150))
enerygy_norm_k_list = []
for k in iteration_k:
    energy_norm_k = 2*pow((((math.sqrt(kappa) -1))/(math.sqrt(kappa)+1)),k)*energy_norm_0
    enerygy_norm_k_list.append(energy_norm_k)

# Convert each error norm to log10 base  
log_energy1 = [math.log10(k) for k in enerygy_norm_k_list]
X_Y_Spline =make_interp_spline(iteration_k,log_energy1)
iteration = np.linspace(0,max(iteration_k),500)
Y = X_Y_Spline(iteration)

plt.plot(iteration,Y,label ='theoretical convergence')
plt.xlabel("iteration")
plt.ylabel("error theoretical log10 base")
plt.title("Relationship between error and iteration")


# our error
our_err = find_error(C,x0,b1,1e-6)

log_energy2= [math.log10(j) for j in our_err]

iteration2 = list(range(0,len(our_err)))
plt.plot(iteration2,log_energy2,label ='our line')
plt.legend()
plt.show()

plt.savefig('errorComparison.pgf')



# different condition number different rate

 # decide the matrix size
n  = np.random.randint(2,100)
# create the matrix

for j in range(1,5):
    R = create_matrix(n)
    KAP = find_cond_num(R)
    b2 = np.random.randint(10,size =(n,1))
    x00 = np.random.randint(10,size =(n,1))
    us_error = find_error(R,x00,b2,1e-6)
    log_energy3= [math.log10(l) for l in us_error]
    iteration3 = list(range(0,len(us_error)))
    plt.plot(iteration3,log_energy3,label =KAP)
    plt.legend()
plt.xlabel("iteration")
plt.ylabel("error log10 base")
plt.title("Different condition number (left corner) causes different convergence rate")
plt.show()
plt.savefig('condtion number comparison.pgf')