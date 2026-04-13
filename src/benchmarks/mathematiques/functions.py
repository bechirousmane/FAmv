"""
CEC 2013 Benchmark Functions Wrapper
===================================

This module provides a lightweight wrapper around the CEC 2013 benchmark
functions implemented in the `opfunu` Python library. The wrapper ensures
compatibility with optimization frameworks where the objective function
is expected to be a callable that takes a position vector and returns
a scalar fitness value.

References
----------
CEC 2013 Benchmark Suite:
    Liang, J. J., Qu, B. Y., Suganthan, P. N., & Hernández-Díaz, A. G. (2013).
    Problem Definitions and Evaluation Criteria for the CEC 2013 Special
    Session on Real-Parameter Optimization.
    Technical Report, Zhengzhou University.

Opfunu Library:
    Van Thieu, N. (2024).
    Opfunu: An open-source Python library for optimization benchmark functions.
    Journal of Open Research Software.

Usage in Mixed-Variable Optimization:
    Wang, F., Zhang, H., & Zhou, A. (2021).
    A particle swarm optimization algorithm for mixed-variable optimization problems.
    Swarm and Evolutionary Computation, 60, 100808.
"""

from opfunu.cec_based.cec2013 import *

class CEC2013Func:
    """
    Callable wrapper for a CEC 2013 benchmark function.

    This class wraps a CEC 2013 function from the `opfunu` library and exposes
    it as a callable object compatible with optimization frameworks that
    require a fitness function of the form:

        f(x) -> float

    The input vector `x` is assumed to already satisfy all domain constraints
    (e.g., continuous bounds, discrete rounding). Therefore, no preprocessing
    or projection is applied inside the benchmark function itself.

    Parameters
    ----------
    func_cls : class
        A CEC 2013 benchmark function class from `opfunu.cec_based.cec2013`,
        such as `F12013`, `F22013`, etc.

    ndim : int
        Dimensionality of the search space.

    Notes
    -----
    - The benchmark implementation is used *as-is*.
    - No modification of the objective function landscape is performed.
    - This design is consistent with the experimental protocol described in
      Wang et al. (2021) for mixed-variable optimization.
    """

    def __init__(self, func_cls, ndim):
        self.func = func_cls(ndim=ndim)
        self.f_global = self.func.f_global

    def __call__(self, x):
        """
        Evaluate the benchmark function at position `x`.

        Parameters
        ----------
        x : list or numpy.ndarray
            Candidate solution vector. The vector must already satisfy the
            domain constraints of the benchmark function.

        Returns
        -------
        float
            Fitness value of the candidate solution.
        """
        return np.abs(self.func.evaluate(x) - self.func.f_global)


NDIM = 50

# Unimodal Functions
F1  = CEC2013Func(F12013,  ndim=NDIM)  # Shifted Sphere
F2  = CEC2013Func(F22013,  ndim=NDIM)  # Shifted Rotated High Conditioned Elliptic
F3  = CEC2013Func(F32013,  ndim=NDIM)  # Shifted Rotated Bent Cigar
F4  = CEC2013Func(F42013,  ndim=NDIM)  # Shifted Rotated Discus
F5  = CEC2013Func(F52013,  ndim=NDIM)  # Shifted Rotated Different Powers

# Multimodal Functions
F6  = CEC2013Func(F62013,  ndim=NDIM)  # Shifted Rosenbrock
F7  = CEC2013Func(F72013,  ndim=NDIM)  # Shifted Rotated Schaffer
F8  = CEC2013Func(F82013,  ndim=NDIM)  # Shifted Rotated Ackley
F9  = CEC2013Func(F92013,  ndim=NDIM)  # Shifted Rotated Weierstrass
F10 = CEC2013Func(F102013, ndim=NDIM)  # Shifted Rotated Griewank
F11 = CEC2013Func(F112013, ndim=NDIM)  # Shifted Rastrigin
F12 = CEC2013Func(F122013, ndim=NDIM)  # Shifted Rotated Rastrigin
F13 = CEC2013Func(F132013, ndim=NDIM)  # Shifted Non-continuous Rastrigin
F14 = CEC2013Func(F142013, ndim=NDIM)  # Shifted Schwefel
F15 = CEC2013Func(F152013, ndim=NDIM)  # Shifted Rotated Schwefel
F16 = CEC2013Func(F162013, ndim=NDIM)  # Shifted Katsuura
F17 = CEC2013Func(F172013, ndim=NDIM)  # Shifted Lunacek Bi-Rastrigin
F18 = CEC2013Func(F182013, ndim=NDIM)  # Shifted Rotated Lunacek Bi-Rastrigin
F19 = CEC2013Func(F192013, ndim=NDIM)  # Shifted Expanded Griewank + Rosenbrock
F20 = CEC2013Func(F202013, ndim=NDIM)  # Shifted Expanded Scaffer F6


# Composition Functions
F21 = CEC2013Func(F212013, ndim=NDIM)  # Composition Function 1
F22 = CEC2013Func(F222013, ndim=NDIM)  # Composition Function 2
F23 = CEC2013Func(F232013, ndim=NDIM)  # Composition Function 3
F24 = CEC2013Func(F242013, ndim=NDIM)  # Composition Function 4
F25 = CEC2013Func(F252013, ndim=NDIM)  # Composition Function 5
F26 = CEC2013Func(F262013, ndim=NDIM)  # Composition Function 6
F27 = CEC2013Func(F272013, ndim=NDIM)  # Composition Function 7
F28 = CEC2013Func(F282013, ndim=NDIM)  # Composition Function 8



if __name__ == "__main__":
    # Plot F1 in 3D
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(-100, 100, 100)
    Y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([
    [F3([x, y]) for x, y in zip(X_row, Y_row)]
    for X_row, Y_row in zip(X, Y)
    ])

    ax.plot_surface(X, Y, Z, cmap="viridis")
    plt.show()




