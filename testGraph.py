from CG import CG, is_pos_def, bounds,cond_size,create_matrix,find_cond_num,cond_size,find_error
import numpy as np
import math
import matplotlib.pyplot as plt
# Setting up:
n =10
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
size_list = range(2,50)
cond_list = []
for n in range(2,50):
    M = create_matrix(n)
    M_cond = find_cond_num(M)
    cond_list.append(M_cond)

plt.figure(figsize=(10, 5))
# calling the function
cond_size(cond_list, size_list)

plt.plot(cond_list, size_list , label ="line 1")
plt.xlabel("kappa the condition number")
plt.ylabel("n the size of the matrix")
plt.title("Relationship between n and kappa")
plt.legend()



# Theory error plotting
C_x_0 = C.dot(x0) 
energy_norm_0 = math.sqrt(np.dot(x0.T,C_x_0)) #get A norm of error of initial guess
iteration_k = np.linspace(0,10,10)
energy_norm_k = 2*pow((((math.sqrt(kappa) -1))/(math.sqrt(kappa)+1)),iteration_k)*energy_norm_0






fig, ax = plt.subplots()
plt.plot(iteration_k,energy_norm_k)
plt.xlabel("iteration")
plt.ylabel("error theoretical")
plt.title("Relationship error and iteration")



# our error plotting

our_err = find_error(C,x0,b1,1e-6)
plt.plot(our_err,'ro')
plt.show()