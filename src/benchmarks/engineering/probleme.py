import math
from math import sqrt
import numpy as np


def Vessel(x):
    """
    Pressure vesssel design
    x1: thickness (d1)  --> discrete value multiple of 0.0625 in 
    x2: thickness of the heads (d1) ---> discrete value multiple of 0.0625 in
    x3: inner radius (r)  ---> cont. value between [10, 200]
    x4: length (L)  ---> cont. value between [10, 200]
    """
    x = [x[2], x[3], x[0], x[1]]  # Reorder to match the original variable names

    y = 0.6224*x[0]*x[2]*x[3]+1.7781*x[1]*x[2]**2+3.1661*x[0]**2*x[3]+19.84*x[0]**2*x[2]

    g1 = -x[0]+0.0193*x[2]
    g2 = -x[1]+0.00954*x[2]
    g3 = -math.pi*x[2]**2*x[3]-(4/3)*math.pi*x[2]**3 + 1296000
    g4 = x[3]-240
    g=[g1,g2,g3,g4]
    
    phi=sum(max(item,0) for item in g)
    eps=1e-5 #tolerance to escape the constraint region
    penality=1e6 #large penality to add if constraints are violated
    
    if phi > eps:  
        fitness=phi+penality
    else:
        fitness=y
    return fitness

def BEAM(x):
    try:
        y = 1.10471*x[0]**2*x[1]+0.04811*x[2]*x[3]*(14.0+x[1])

        # parameters
        P = 6000; L = 14; E = 30e+6; G = 12e+6
        t_max = 13600; s_max = 30000; d_max = 0.25

        M = P*(L+x[1]/2)
        R = sqrt(0.25*(x[1]**2+(x[0]+x[2])**2))
        J = 2*(sqrt(2)*x[0]*x[1]*(x[1]**2/12+0.25*(x[0]+x[2])**2))
        P_c = (4.013*E/(6*L**2))*x[2]*x[3]**3*(1-0.25*x[2]*sqrt(E/G)/L)
        t1 = P/(sqrt(2)*x[0]*x[1]); t2 = M*R/J
        t = sqrt(t1**2+t1*t2*x[1]/R+t2**2)
        s = 6*P*L/(x[3]*x[2]**2)
        d = 4*P*L**3/(E*x[3]*x[2]**3)
        # Constraints
        g1 = t-t_max; #done
        g2 = s-s_max; #done
        g3 = x[0]-x[3]
        g4 = 0.10471*x[0]**2+0.04811*x[2]*x[3]*(14.0+x[1])-5.0
        g5 = 0.125-x[0]
        g6 = d-d_max
        g7 = P-P_c; #done

        g=[g1,g2,g3,g4,g5,g6,g7]
        g_round=np.round(np.array(g),6)
        w1=100
        w2=100

        phi=sum(max(item,0) for item in g_round)
        viol=sum(float(num) > 0 for num in g_round)

        reward = (y + (w1*phi + w2*viol))
    except ZeroDivisionError as e:
        print(f"Error in BEAM evaluation: {e}")
        reward = float('inf')  # Return a large penalty if division by zero occurs

    return reward

import numpy as np

def CSD(x):
    """
    Coil Spring Design benchmark problem.

    Parameters
    ----------
    x : array-like
        Design vector [d, D, N]

    Returns
    -------
    float
        Penalized objective value
    """
    penalty=1e6
    d, D, N = x

    # Constants
    P_max = 1000.0
    S = 189e3
    G = 11.5e6
    L_free = 14.0
    d_min = 0.2
    D_max = 3.0
    P_load = 300.0
    delta_pm = 6.0
    delta_w = 1.25

    try:
        # Objective function 
        f = (N + 2) * d**2 * D

        # Mechanical coefficients
        C_f = (4 * (D/d) - 1) / (4 * (D/d) - 4) + (0.615 * d) / D
        K = (G * d**4) / (8 * N * D**3)

        # Inequality constraints g_i(x) <= 0
        delta_max = P_max / K
        delta_load = P_load / K

        constraints = [
            (8 * C_f * P_max * D) / (np.pi * d**3) - S,          
            delta_max + 1.05 * (N + 2) * d - L_free,              
            d_min - d,                                             
            (d + D) - D_max,                                       
            3 - D / d,                                            
            delta_max - delta_pm,                                
            delta_w - delta_max + delta_load,                      
        ]

    except ZeroDivisionError as e:
        print(f"Error in CSD evaluation: {e}")
        return float('inf')  # Return a large penalty if division by zero occurs

    # Penalty for violations
    violation = sum(max(0.0, g) for g in constraints)

    return f + penalty * violation