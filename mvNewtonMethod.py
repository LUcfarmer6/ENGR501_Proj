# --------------------------------------------------------------
#    Numerical Methods ISV Model Using Newton-Raphson Method
# --------------------------------------------------------------

# Description:
# This model utilizes a set of material constants (C_n through C_m)
# along with inputs of strain rate and time step to calculate and predict
# an accurate stress-strain curve for that given material. This is a
# viscoplastic model (strain-rate sensitive) that uses a Radial-Return
# scheme to predict deformation elastically, and then correct for any plastic
# deformation that also occurs. Plastic deformation behavior is governed by
# the Flow Rule and Isotropic Hardening. These equations may be solved either
# analytically or numerically (using the Newton-Raphson (N-R) method). The analytical
# method will be solved first, requiring a much smaller time step for
# accuracy, while the N-R method is solved second which retains accuracy at
# much greater time steps. The two methods are then compared and tested to
# optimize time step, computational time, and accuracy of the resulting curve.

## Import Packages
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import timeit
from matplotlib import pyplot as plt
## Define System of Equations
# INPUT: Array of independent variables x = [x0,x1,...,xn]
#   Inputs will be strain rate (eps_dot), time step (t_step), and temperature (T)
#   along with other material constants (C_1 to C_n)
eps_dot = 0.1        # Strain Rate in Units [1/s]
perc_elong = 11     # percent elongation * 100
N_partitions = 200
time_step = perc_elong/eps_dot/100/N_partitions   # Time step [seconds]
T = 296             # Temperature [Kelvin] (23°C)
C = np.ones(19)
# sigma_n = jnp.zeros(N_partitions)           # Initial value of stress (should be zero)
sigma_n = [0.0]
kappa_n = [0.0]
delta_eps_p_n = [0.0]
# kappa_n = jnp.zeros(N_partitions)           # Initial value of kappa
# delta_eps_p_n = jnp.zeros(N_partitions)     # Initial guess for plastic strain
mu = 1e3                                      # Shear modulus [MPa] (guess value)
C[1] = 312.86e6         # MPa
C[2] = 154.78           # K
C[3] = 27.2e6           # MPa
C[4] = 818.26           # K
C[5] = 6914.1           # 1/s
C[6] = 233.39           # K
C[7] = 9e-6             # 1/MPa
C[8] = 1632.34          # K
C[9] = 148.36e6         # MPa
C[10] = 942.28          # K
C[11] = 100.67e-6       # s/MPa
C[12] = 2517.12         # K
C[13] = 98.53e-6        # 1/MPa
C[14] = 171.56          # K
C[15] = 8950.63e6       # MPa
C[16] = 279.18          # K
C[17] = 7363.75e-6      # s/MPa
C[18] = 3316.82         # K

#   Equation constants
# P = pressure (Mega-Pascals)
# T = temperature (Kelvin)
# s = time (seconds)
V = C[1] * jnp.exp(-C[2] / T)       # P*e^(T/T)
f = C[5] * jnp.exp(-C[6] / T)       # (1/s)*e^(T/T)
R_d = C[13] * jnp.exp(-C[14] / T)   # (1/P)*e^(T/T)
# (line 122) shows H must be MPa
H = C[15] - C[16] * T               # P - T^2 
R_s = C[17] * jnp.exp(-C[18] / T)   # (s/P)*e^(T/T)

# OUTPUT: Array of dependent Results res = [f0, f1, ..., fn]
#   Outputs will be arrays of strain and stress data to be plotted

# def fs(x):
#     # From http://fourier.eng.hmc.edu/e176/lectures/NM/node21.html Example 1
#     # f0 = 3*x_0 - cos(x_1*x_2)--1.5
#     # f1 = 4*x_0^2-625*x_1^2+2*x-1
#     # f2 = 20*x_2+exp(-x_0*x_1)+9
#     f0 = 3*x[0]-jnp.cos(x[1]*x[2])-3/2  # Equation 1
#     f1 = 4*x[0]**2-625*x[1]**2+2*x[2]-1 # Equation 2
#     f2 = 20*x[2]+jnp.exp(-x[0]*x[1])+9  # Equation 3
#     res = [f0, f1, f2]                  # Create Array of Results
#     return jnp.asarray(res)             # Return Jax Array

def radialReturn(sigma_n, delta_eps_p_n, kappa_n):          # Defines function inputs from main logical loop
    # sigma_tr_n: trial stress from current time iteration
    # kappa_n: kappa from current time iteration
    # kappa_n1: guess for kappa at next timestep iteration

    # First making Elastic prediction
    # Elastic strain increment
    delta_eps_e = eps_dot*time_step             # eps = (eps_dot)*(time)
    # Hooke's Law (2*mu = E)
    delta_sigma_tr = 2*mu*delta_eps_e           # MPa   <-- this come from mu (line 42) !!
    # Stress at next iteration is equal to stress at current iteration plus a change in stress
    sigma_tr = sigma_n + delta_sigma_tr         # MPa
    print(sigma_tr)
    # ___ = ___ - [(1/P)(eps) + (s/P)(s)](___)^2
    # kappa needs to evaluate to (MPa) for Y_f (line 107) !!
    # kappa_tr = kappa_n-(R_d*delta_eps_p_n+R_s*time_step)*kappa_n**2
    kappa_tr = kappa_n-(R_d*delta_eps_e+R_s*time_step)*kappa_n**2
    # beta = V * jnp.arcsinh(delta_eps_p_n/time_step/f)         # MPa
    beta = V * jnp.arcsinh(eps_dot/f)       # MPa[sin((1/s)/(1/s))]
    # Yield function should be a function of stress, ISV's, and strain rate
    Y_f = sigma_tr-kappa_tr-beta        # MPa

    if Y_f <= 0:                                # If less than zero, deformation is purely elastic
        # Stress at next iteration is equal to stress at current iteration plus a change in stress
        sigma_n1 = sigma_tr             # MPa
        # Total strain is equal to the elastic strain
        delta_eps = delta_eps_e         # eps
        kappa_n1 = kappa_tr             # MPa
        delta_eps_p_n1 = 0              # eps
        print("ELASTIC")
    else:                                       # If yield function greater than zero, plasticity occurs. Must solve for plastic strain numerically
        # (reference the N-R method function here so solve plastic strain)
        # initial guess for plastic strain is previous plastic strain
        delta_eps_p_n1 = delta_eps_p_n      # eps/s
        # initial guess of future kappa with initial guess of future plastic strain
        kappa_n1 = kappa_tr + H * delta_eps_p_n1        # must be MPa <-- H must be MPa
        xs = [delta_eps_p_n1, kappa_n1]    # vector of future guesses
        def fs(x):
            result = plasticity(x, sigma_tr, kappa_tr, delta_eps_p_n, kappa_n)  # input "xs" returns array "fs"
            return result
        res = multivariateNewton(fs, xs, 1e-5, 30) # Perform Newton Method for System "fs" with guess  [x0,x1,x2] = [1,1,1] with tol = 1e-8 and N maximum iterations
        # print(fs(res))                  # Print "fs" output for system
        # proclaims future delta_eps_p from Newton Raphson
        delta_eps_p_n1 = res[0]         # eps
        # proclaims future stress
        sigma_n1 = sigma_tr - 2 * mu * delta_eps_p_n1       # MPa
        # print(sigma_n1)
        # proclaims future kappa from Newton Raphson's F2
        kappa_n1 = res[1]               # MPa
    return [sigma_n1, kappa_n1, delta_eps_p_n1]

def plasticity(x, sigma_tr, kappa_tr, kappa_n, delta_eps_P_n):  
# solves using yield function and isotropic hardening equation
    delta_eps_p = x[0]      # eps
    kappa_n1 = x[1]         # MPa
    beta = V * jnp.arcsinh(delta_eps_p/time_step/f)         # MPa[sin((1/s)/(1/s))]
    # yield function
    F1 = sigma_tr-kappa_tr-beta         # MPa
    # P - [(1/P) + (s2/P)](P2) - P
    # R_s can't be s^2 !!! (look back @ line 71)
    F2 = kappa_n-(R_d*delta_eps_p+R_s*time_step)*kappa_n1**2-kappa_n1
    fs = jnp.asarray([F1, F2])  # Write outputs to function array for logical loop
    return fs


## Define Multivariate Newton Method
# INPUT: System of Functions = f, Initial Guess Array = x0, tolerance = tol, Maximum iterations = N
# OUTPUT: Solution Array = x
def multivariateNewton(f, x0, tol, N):
    x0 = jnp.asarray(x0).T          # Convert Input Array to Jax Array
    def J_inv(x):                   # Create Inverse Jacobian Function
        jacobian = jax.jacfwd(f)    # Calculate the jacobian function from the provided systems with Forward Auto-differentiation
        J = jacobian(x)  # Calculate the Jacobian at x
        J_inv = jnp.linalg.inv(J)   # Calculate the Inverse Jacobian
        return jnp.asarray(J_inv)   # Return Inverse Jacobian at x as a Jax Array
    for k in range(1,N):            # Start Loop for Maximum Iterations
        x = jnp.subtract(x0, jnp.matmul(J_inv(x0), f(x0).T)) # Perform Newton Iteration: x_{n+1} = x_n-J^(-1)*f
        # reltol = jnp.divide(jnp.linalg.norm(jnp.subtract(x,x0), np.inf),jnp.linalg.norm(x, np.inf)) # Calculate: ||x_{n+1}-x_n||/||x_{n+1}||
        atol = jnp.linalg.norm(jnp.subtract(x,x0), np.inf) # Calculate: ||x_{n+1}-x_n||/||x_{n+1}||
        # print(i, tol)             # Print iteration and relTol
        if atol < tol:              # Check for convergence
            # print(x)              # Print Result
            return x                # Return Result
        x0 = x                      # Update x0 for Next iteration
    print("Failed to converge")     # Print Message if Convergence did not occur


## Main Logical Loop to build Stress-Strain curve
for n in range(0,N_partitions):     # nth timestep partition of strain subdivisions
    # sigma_n = sigma_n1            # makes current stress from previous future stress
    # delta_eps_p_n = delta_eps_p_n1
    # delta_eps_p_k, initially 0, should be output at end of current iteration for future iteration
    # kappa_n = kappa_n1            # makes current kappa from previous future kappa
    # print(sigma_n[n])
    sigma_n1, kappa_n1, delta_eps_p_n1 = radialReturn(sigma_n[-1], delta_eps_p_n[n], kappa_n[n])  # calls function with current stress and kappa
    # jax.ops.index_update(sigma_n, jax.ops.index[n+1],sigma_n1)
    sigma_n.append(sigma_n1)
    # print(delta_eps_p_n[n+1])
    delta_eps_p_n.append(delta_eps_p_n1)
    # jax.ops.index_update(delta_eps_p_n, jax.ops.index[n+1],delta_eps_p_n1)
    kappa_n.append(kappa_n1)
    # jax.ops.index_update(kappa_n, jax.ops.index[n + 1], kappa_n1)
    print(sigma_n)
    # print(delta_eps_p_n)
    print(n)

## Plot Outputs to show Stress-Strain curve
# plot stress-strain curve
# converts x-axis from partitions to total strain applied
plt.plot(time_step*range(0, N_partitions + 1)/eps_dot, sigma_n)
plt.show()



## End of Document
# that's all folks!