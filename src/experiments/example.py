"""
Simple example demonstrating the Firefly Algorithm on benchmark functions.
This example shows how to use the FA library with both standard and mixed-variable algorithms.
"""

import os
import numpy as np
from benchmarks.mathematiques.functions import F1
from search_space.search_space import SearchSpace
from search_space.dimension import ContinuousDimension, DiscreteDimension
from optimization.probleme import Probleme
from optimization.fa.fa import FA, FA_Hamming_mv

# Configuration
SEED = 42
LOWER_BOUND = -100
UPPER_BOUND = 100

# Search space dimensions
CONTINUOUS_DIM = 25  # Number of continuous dimensions
DISCRETE_DIM = 25    # Number of discrete dimensions

# Algorithm parameters
POPULATION_SIZE = 25
MAX_EVALUATIONS = 1e5
DURATION = 600  # Maximum time in seconds
NUM_ITERATIONS = 3  # Number of independent runs

# Algorithm hyperparameters
ALPHA = 1.5   # Light absorption coefficient
BETA0 = 1.5   # Attractiveness at distance 0
GAMMA = 0.1   # Randomness parameter


def create_search_space(seed=SEED):
    """
    Create a mixed search space with continuous and discrete dimensions.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        SearchSpace object with all dimensions
    """
    # Create continuous dimensions
    continuous_dims = [
        ContinuousDimension(lower=LOWER_BOUND, upper=UPPER_BOUND, name="")
        for _ in range(CONTINUOUS_DIM)
    ]
    
    # Create discrete dimensions with clipping projection
    discrete_dims = [
        DiscreteDimension(
            values=list(range(LOWER_BOUND, UPPER_BOUND)),
            projection_rules=lambda x: np.clip(x, LOWER_BOUND, UPPER_BOUND),
            name=""
        )
        for _ in range(DISCRETE_DIM)
    ]
    
    # Combine all dimensions
    all_dims = continuous_dims + discrete_dims
    return SearchSpace(dimensions=all_dims, seed=seed)


def create_optimization_problem(search_space, fitness_func):
    """
    Create an optimization problem with proper dimension indexing.
    
    Args:
        search_space: SearchSpace object
        fitness_func: Objective function to minimize
        
    Returns:
        Probleme object
    """
    # Define which dimensions are continuous and which are discrete
    continuous_idx = list(range(CONTINUOUS_DIM))
    discrete_idx = list(range(CONTINUOUS_DIM, CONTINUOUS_DIM + DISCRETE_DIM))
    
    return Probleme(
        search_space=search_space,
        continuous_idx=continuous_idx,
        discrete_idx=discrete_idx,
        fitness_func=fitness_func,
        population_size=POPULATION_SIZE
    )


def run_standard_fa(fitness_func, seed=SEED):
    """
    Run the standard Firefly Algorithm on continuous and discrete spaces.
    
    Args:
        fitness_func: Objective function to minimize
        seed: Random seed
        
    Returns:
        FA instance after execution
    """
    search_space = create_search_space(seed)
    problem = create_optimization_problem(search_space, fitness_func)
    
    # Create and configure FA
    fa = FA(
        probleme=problem,
        alpha=ALPHA,
        beta0=BETA0,
        gamma=GAMMA,
        max_evaluations=MAX_EVALUATIONS,
        duration=DURATION,
        seed=seed
    )
    
    # Execute optimization
    print("Running Standard FA...")
    fa.run(verbose=True)
    
    return fa


def run_hamming_fa(fitness_func, seed=SEED):
    """
    Run FA with Hamming distance for mixed-variable optimization.
    Uses Hamming distance for discrete dimensions and Euclidean for continuous.
    
    Args:
        fitness_func: Objective function to minimize
        seed: Random seed
        
    Returns:
        FA_Hamming_mv instance after execution
    """
    search_space = create_search_space(seed)
    problem = create_optimization_problem(search_space, fitness_func)
    
    # Create and configure FA with Hamming distance
    fa_hamming = FA_Hamming_mv(
        probleme=problem,
        alpha=ALPHA,
        alpha_d=ALPHA,  # Separate alpha for discrete dimensions
        beta0=BETA0,
        gamma=GAMMA,
        int_val=True,
        max_evaluations=MAX_EVALUATIONS,
        duration=DURATION,
        seed=seed
    )
    
    # Execute optimization
    print("Running FA with Hamming Distance")
    fa_hamming.run(verbose=True)
    
    return fa_hamming


def run_example():
    """
    Main example: optimize F1 (Sphere function) using both FA variants.
    """
    print("Firefly Algorithm Example - F1 Sphere Function Optimization")
    
    # Storage for results
    fa_histories = []
    hamming_histories = []
    
    # Run multiple iterations for statistical analysis
    for iteration in range(NUM_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1}/{NUM_ITERATIONS} ---\n")
        
        # Run standard FA
        fa = run_standard_fa(F1, seed=SEED + iteration)
        fa_histories.append(fa.best_fitness_historie)
        
        # Run Hamming FA
        fa_hamming = run_hamming_fa(F1, seed=SEED + iteration)
        hamming_histories.append(fa_hamming.best_fitness_historie)
    
    # Print summary statistics
    fa_final_values = [h[-1] for h in fa_histories]
    hamming_final_values = [h[-1] for h in hamming_histories]
    
    print("Summary Statistics")
    print(f"Standard FA - Mean: {np.mean(fa_final_values):.6e}, Std: {np.std(fa_final_values):.6e}")
    print(f"Hamming FA  - Mean: {np.mean(hamming_final_values):.6e}, Std: {np.std(hamming_final_values):.6e}")


if __name__ == "__main__":
    run_example()
