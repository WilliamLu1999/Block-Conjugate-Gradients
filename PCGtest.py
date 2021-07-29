from PCG import PCG, timing_PCG
from CG import CG,find_cond_num
import numpy as np
import scipy.sparse.linalg
import math
import matplotlib.pyplot as plt

####################################################################
# number of iterations vary everytime as matrix b are random
####################################################################
# Testing matrix of size 882. First parameter.
A1 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/A_n_882_gamma_0.npz')
Minv1 =scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/P_n_882_gamma_0.npz')

b = np.random.randint(5,size =(A1.shape[0],1))
x = np.zeros((A1.shape[0],1))
xr = scipy.sparse.linalg.spsolve(A1,b) # real solution
xr = np.reshape(xr,(A1.shape[0],1)) # make the dimension correct
####################################################################

eigens2 = scipy.sparse.linalg.eigsh(Minv1@A1,k = A1.shape[0]-1)
#indexEigen = np.asarray(list(range(0,len(eigenVals))))
#eigenVals = np.asarray(eigens,dtype= float)

####################################################################
# Scipy's PCG error line
history=[]
history_error =[]
def report(xk):
    history.append(xk.copy())
    return history
x_sci = scipy.sparse.linalg.cg(A1,b,x,1e-6,callback=report,M=Minv1)
history_arr =np.array(history)
history_arr_T = history_arr.T
xr_expand = np.tile(xr.T,(len(history),1))

error_Matrix =  history_arr - xr_expand
error_vector = np.split(error_Matrix,len(history))

for i in error_vector:
    #xr is the real solution. errSci is the error of each iteration xi
    i_T = i.T
    i_A1_T = A1.dot(i_T)
    energy_norm_i = math.sqrt(np.dot(i_T.T,i_A1_T))
    history_error.append(energy_norm_i)
    log_energy_sci= [math.log10(j) for j in history_error]
iteration_sci = list(range(0,len(history_error)))



sol_PCG_1 = PCG(A1,x,b,Minv1,xr,1e-6,False) # 91 iterations according to the terminal
sol_CG_1 = CG(A1,x,b,xr,1e-6,False) #211
log_pcg_1 = [math.log10(i) for i in sol_PCG_1]
iteration_pcg_1 = list(range(0,len(log_pcg_1)))
MinvA_1 = (Minv1@A1).toarray()
pcg_kappa_1 = find_cond_num(MinvA_1) # see how condition number plays a role after preconditioner

###############################################################
# Testing matrix with of 882. Second parameter.
A2 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/A_n_882_gamma_1.npz')
Minv2 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/P_n_882_gamma_1.npz')
xr2 = scipy.sparse.linalg.spsolve(A2,b)
xr2 = np.reshape(xr2,(A2.shape[0],1))
sol_PCG_2 = PCG(A2,x,b,Minv2,xr2,1e-6,False) #98
sol_CG_2 = CG(A2,x,b,xr2,1e-6,False)  #197

###############################################################
# Testing matrix with of 882. Third parameter.
A3 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/A_n_882_gamma_1000.npz')
Minv3 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/P_n_882_gamma_1000.npz')
xr3 = scipy.sparse.linalg.spsolve(A3,b)
xr3 = np.reshape(xr3,(A3.shape[0],1))
sol_PCG_3 = PCG(A3,x,b,Minv3,xr3,1e-6,False) #226
sol_CG_3 = CG(A3,x,b,xr3,1e-6,False)  #1035

###############################################################
# Testing matrix with of 3362. First parameter.
A4 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/A_n_3362_gamma_0.npz')
Minv4 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/P_n_3362_gamma_0.npz')
xx = np.zeros((A4.shape[0],1)) # new initial guess for the following 3 matrices
bb = np.random.randint(5,size =(A4.shape[0],1)) # new right hand side for the following 3 matrices
xr4 =scipy.sparse.linalg.spsolve(A4,bb)
xr4 =np.reshape(xr4,(A4.shape[0],1))
sol_PCG_4 = PCG(A4,xx,bb,Minv4,xr4,1e-6,False) #173
sol_CG_4 = CG(A4,xx,bb,xr4,1e-6,False)  #408

###############################################################
# Testing matrix with of 3362. Second parameter.
A5 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/A_n_3362_gamma_1.npz')
Minv5 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/P_n_3362_gamma_1.npz')
xr5 =scipy.sparse.linalg.spsolve(A5,bb)
xr5 =np.reshape(xr5,(A5.shape[0],1))
sol_PCG_5 = PCG(A5,xx,bb,Minv5,xr5,1e-6,False) #197
sol_CG_5 = CG(A5,xx,bb,xr5,1e-6,False)  #379

###############################################################
# Testing matrix with of 3362. Third parameter.
A6 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/A_n_3362_gamma_1000.npz')
Minv6 = scipy.sparse.load_npz('/Users/William/Downloads/SparseMatrices/P_n_3362_gamma_1000.npz')
xr6 = scipy.sparse.linalg.spsolve(A6,bb)
xr6 = np.reshape(xr6,(A6.shape[0],1))
sol_PCG_6 = PCG(A6,xx,bb,Minv6,xr6,1e-6,False) #705
sol_CG_6 = CG(A6,xx,bb,xr6,1e-6,False)  #3081

###############################################################
# First graph: size 882. CG vs PCG.
plt.xlabel("iteration")
plt.ylabel("error theoretical log10 base")
plt.title("Relationship between error and iteration")
plt.plot(iteration_pcg_1,log_pcg_1,label =('PCG, n=882, \u03B3=0, \u03BA=%1.0f'%round(pcg_kappa_1.real,2)))
log_cg_1 = [math.log10(j) for j in sol_CG_1]
iteration_cg_1 = list(range(0,len(log_cg_1)))
cg_kappa_1 = find_cond_num(A1.toarray())
plt.plot(iteration_cg_1,log_cg_1,label =('CG, n=882, \u03B3=0, \u03BA=%1.0f'%round(cg_kappa_1.real,2)))
plt.plot(iteration_sci,log_energy_sci,label =('Scipy PCG, n=882, \u03B3=0, \u03BA=%1.0f'%round(cg_kappa_1.real,2)))
plt.legend()
plt.show() # First graph: 882 CG vs PCG. Gamma 0. Compare condition.
###############################################################
# second graph: Fixed Condition number, 3 different gamma. same size 882. PCG only.
plt.xlabel("iteration")
plt.ylabel("error theoretical log10 base")
plt.title("How different parameters/preconditioner influence convergence")
plt.plot(iteration_pcg_1,log_pcg_1,label =('PCG, n=882, \u03B3=0, \u03BA=%1.0f'%round(pcg_kappa_1.real,2)))
plt.legend()
log_pcg_2 = [math.log10(i) for i in sol_PCG_2]
iteration_pcg_2 = list(range(0,len(log_pcg_2)))
MinvA_2 = (Minv2@A2).toarray()
pcg_kappa_2 = find_cond_num(MinvA_2)
plt.plot(iteration_pcg_2,log_pcg_2,label =('PCG, n=882, \u03B3=1, \u03BA=%1.0f'%round(pcg_kappa_2.real,2)))
plt.legend()
log_pcg_3 = [math.log10(i) for i in sol_PCG_3]
iteration_pcg_3 = list(range(0,len(log_pcg_3)))
MinvA_3 = (Minv3@A3).toarray()
pcg_kappa_3 = find_cond_num(MinvA_3)
plt.plot(iteration_pcg_3,log_pcg_3,label =('PCG, n=882, \u03B3=1000, \u03BA=%1.0f'%round(pcg_kappa_3.real,2)))
plt.legend()
plt.show()
###############################################################

# Third graph: see how matrix size play a role. PCG only. gamma-0.
plt.xlabel("iteration")
plt.ylabel("error theoretical log10 base")
plt.title("same parameter but different matrix size")
#plt.plot(iteration_pcg_1,log_pcg_1,label =('PCG-0',round(pcg_kappa_1.real,2)))
plt.legend()
log_pcg_4 = [math.log10(i) for i in sol_PCG_4]
iteration_pcg_4 = list(range(0,len(log_pcg_4)))
MinvA_4 = (Minv4@A4).toarray()
pcg_kappa_4 = find_cond_num(MinvA_4)
#plt.plot(iteration_pcg_4,log_pcg_4,label =('PCG-0',round(pcg_kappa_4.real,2)))
plt.legend()
#plt.show()

###############################################################
# Third graph: how eigenvalues play a role. PCG only. size 3226. Three gamma.
plt.xlabel("iteration")
plt.ylabel("error theoretical log10 base")
plt.title("CG & PCG")
log_cg_4 = [math.log10(i) for i in sol_CG_4]
iteration_cg_4 = list(range(0,len(log_cg_4)))
cg_kappa_4 = find_cond_num(A4.toarray())
#plt.plot(iteration_cg_4,log_cg_4,label =('CG, n=3362, \u03B3=0, \u03BA=%1.0f'%round(cg_kappa_4.real,2))) # the fourth matrix. size 3362,CG-0-4
#plt.legend()
log_cg_5 = [math.log10(i) for i in sol_CG_5]
iteration_cg_5 = list(range(0,len(log_cg_5)))
cg_kappa_5 = find_cond_num(A5.toarray())
plt.plot(iteration_cg_5,log_cg_5,label =('CG, n=3362,\u03BA=%1.0f'%round(cg_kappa_5.real,2))) # \u03B3=1, 
plt.legend()
log_cg_6 = [math.log10(i) for i in sol_CG_6]
iteration_cg_6 = list(range(0,len(log_cg_6)))
cg_kappa_6 = find_cond_num(A6.toarray())
#plt.plot(iteration_cg_6,log_cg_6,label =('CG, n=3362, \u03B3=1000, \u03BA=%1.0f'%round(cg_kappa_6.real,2)))
#plt.legend()
#plt.plot(iteration_pcg_4,log_pcg_4,label =('PCG, n=3362, \u03B3=0, \u03BA=%1.0f'%round(pcg_kappa_4.real,2)))
#plt.legend()
log_pcg_5 = [math.log10(i) for i in sol_PCG_5]
iteration_pcg_5 = list(range(0,len(log_pcg_5)))
MinvA_5 = (Minv5@A5).toarray()
pcg_kappa_5 = find_cond_num(MinvA_5)
plt.plot(iteration_pcg_5,log_pcg_5,label =('PCG, n=3362, \u03BA=%1.0f'%round(pcg_kappa_5.real,2))) #\u03B3=1,
plt.legend()
log_pcg_6 = [math.log10(i) for i in sol_PCG_6]
iteration_pcg_6 = list(range(0,len(log_pcg_6)))
MinvA_6 = (Minv6@A6).toarray()
pcg_kappa_6 = find_cond_num(MinvA_6)
#plt.plot(iteration_pcg_6,log_pcg_6,label =('PCG, n=3362, \u03B3=1000, \u03BA=%1.0f'%round(pcg_kappa_6.real,2)))
#plt.legend()
plt.savefig("11a.png")
plt.close()
##################################################
# scatter plot for preconditioners
'''
plt.xlabel("real part of eigenvalues")
plt.ylabel("imagary pary of eigenvalues")
plt.title("Eigenvalues line distribution")
eigens3 = scipy.sparse.linalg.eigsh(Minv2@A2,k = A2.shape[0]-1)
eigens4 = scipy.sparse.linalg.eigsh(Minv3@A3,k = A3.shape[0]-1)
plt.scatter(eigens2[0].real,eigens2[0].imag,s= 0.1,label ='eigenVals of A with a precond-0')
plt.legend()
plt.scatter(eigens3[0].real,eigens3[0].imag,s= 0.1,label ='eigenVals of A with a precond-1')
plt.legend()
plt.scatter(eigens4[0].real,eigens4[0].imag,s= 0.1,label ='eigenVals of A with a precond-1000')
plt.legend()
plt.show()
'''
#eigens1 = scipy.sparse.linalg.eigsh(A1,k = A1.shape[0]-1)
#eigens3 = scipy.sparse.linalg.eigsh(Minv2@A2,k = A2.shape[0]-1)
#eigens4 = scipy.sparse.linalg.eigsh(Minv3@A3,k = A3.shape[0]-1)
eigens5_cg = scipy.sparse.linalg.eigsh(A5,k = A5.shape[0]-1)
eigens5_pcg = scipy.sparse.linalg.eigsh(Minv5@A5,k = A5.shape[0]-1)
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)  #it should be 4 later
axs = gs.subplots(sharex=True, sharey=True)
fig.suptitle('Spread of Eigenvalues')
axs[0].plot(eigens5_cg[0].real,eigens5_cg[0].imag,'+')
axs[1].plot(eigens5_pcg[0].real, eigens5_pcg[0].imag,'+',color='orange')
#axs[2].plot(eigens3[0].real, eigens3[0].imag,'+',color ='purple')
#axs[3].plot(eigens4[0], eigens4[0].imag, '+',color='brown')
for ax in axs:
    ax.label_outer()
plt.savefig("12a.png")
plt.close()