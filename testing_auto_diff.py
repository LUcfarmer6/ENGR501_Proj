import auto_diff
import numpy as np

def fs(x):
    # From http://fourier.eng.hmc.edu/e176/lectures/NM/node21.html Example 1
    # f0 = 3*x_0 - cos(x_1*x_2)--1.5
    # f1 = 4*x_0^2-625*x_1^2+2*x-1
    # f2 = 20*x_2+exp(-x_0*x_1)+9
    f0 = 3*x[0]-np.cos(x[1]*x[2])-3/2  # Equation 1
    f1 = 4*x[0]**2-625*x[1]**2+2*x[2]-1 # Equation 2
    f2 = 20*x[2]+np.exp(-x[0]*x[1])+9  # Equation 3
    res = [f0, f1, f2]                  # Create Array of Results
    return res             # Return Jax Array
x = [0,0,0]

with auto_diff.AutoDiff(x) as x:
    f_eval = fs(x)
    y, Jf = auto_diff.get_value_and_jacobian(f_eval)
# print(auto_diff.jacobian(fs))