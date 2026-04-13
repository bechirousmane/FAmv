# Firefly Algorithm for Mixed-Variable Optimization

A Python implementation of the Firefly Algorithm (FA) adapted for mixed-variable optimization problems that combine continuous, ordinal, and categorical decision variables.

## Overview

This repository contains the implementation of FAmv (Firefly Algorithm for Mixed-Variable optimization), a metaheuristic algorithm designed to handle optimization problems with heterogeneous variable types. The key innovation is a modified distance-based attractiveness mechanism that integrates both continuous and discrete components into a unified formulation.

### Problem Context

Many real-world optimization problems involve mixed-variable search spaces where:
- **Continuous variables** are real-valued parameters (e.g., temperatures, pressures)
- **Discrete variables** are integer-valued parameters
- **Categorical variables** are non-ordered discrete values

Traditional metaheuristic algorithms are typically designed for either continuous OR discrete optimization, making them less effective for mixed-variable problems. FAmv specifically addresses this challenge.

## What's Included

This implementation provides several variants of the Firefly Algorithm:

1. **Basic FA**: Standard firefly algorithm for continuous optimization
2. **FA-Hamming**: FA with Hamming distance metric for mixed variables
3. **FA-Gower**: FA with Gower distance metric for mixed variables
4. **Adaptive variants**: Versions with adaptive alpha and gamma parameters for improved convergence

## Key Features

- Support for mixed continuous and discrete variables
- Multiple distance metrics (Hamming, Gower)
- Adaptive parameter control during optimization
- Multiprocessing support for parallel evaluation
- Convergence history tracking
- Benchmark evaluation on CEC2013 mixed-variable problems

## Project Structure

```
src/
├── benchmarks/          # Benchmark functions for testing
│   ├── mathematiques/   # Mathematical test functions (F1, F2, etc.)
│   └── engineering/     # Engineering design problems
├── optimization/        # Core optimization algorithms
│   ├── fa/             # Firefly Algorithm implementations
│   └── probleme.py     # Problem definition and management
├── search_space/        # Search space representation
│   ├── dimension.py     # Continuous and discrete dimension definitions
│   ├── particle.py      # Firefly particle representation
│   └── search_space.py  # Search space management
└── experiments/        # Example usage and experiments
    └── example.py      # Quick start example
```

## Installation

```bash
# Clone the repository
git clone https://github.com/bechirousmane/Firefly-algorithm-for-mixed-variable-optimization.git
cd Firefly-algorithm-for-mixed-variable-optimization

# Install dependencies
pip install numpy opfunu
```

## Quick Start

```python
from benchmarks.mathematiques.functions import F1
from optimization.fa.fa import FA, FA_Hamming_mv
from search_space.search_space import SearchSpace
from search_space.dimension import ContinuousDimension, DiscreteDimension

# Define the search space
continuous_dims = [ContinuousDimension(lower=-100, upper=100) for _ in range(25)]
discrete_dims = [DiscreteDimension(values=list(range(-100, 100))) for _ in range(25)]
dims = continuous_dims + discrete_dims

# Create search space
search_space = SearchSpace(dimensions=dims, seed=0)

# Run optimization
fa_mv = FA_Hamming_mv(
    problem=problem,
    alpha=1.5,
    beta0=1.5,
    gamma=0.1,
    max_evaluations=100000,
    seed=0
)

fa_mv.run(verbose=True)
print(f"Best fitness: {fa_mv.best_fitness}")
```

See `src/experiments/example.py` for a complete working example.

## Benchmarking

The implementation is tested on the CEC2013 mixed-variable benchmark suite including:
- Unimodal functions (single global optimum)
- Multimodal functions (multiple local optima)
- Composition functions (combinations of other functions)

Run benchmarks with:

```bash
cd src
python experiments/example.py
```

## Performance

The FAmv algorithm demonstrates competitive and often superior performance compared to state-of-the-art mixed-variable optimization methods on benchmark problems and real-world engineering design challenges.

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{bechir2026firefly,
  title={A Firefly Algorithm for Mixed-Variable Optimization Based on Hybrid Distance Modeling},
  author={Bechir, Ousmane Tom and Jos\'e-Garc\'ia, Ad\'an and Chelly Garcia, Zaineb and Sobanski, Vincent and Dhaenens, Clarisse},
  journal={arXiv preprint arXiv:2603.26792},
  year={2026}
}
```

## License

This project is licensed under the Creative Commons Attribution 4.0 International License.

## References

- Paper: https://arxiv.org/abs/2603.26792
- Authors: Ousmane Tom Bechir, Adán José-García, Zaineb Chelly Dagdia, Vincent Sobanski, Clarisse Dhaenens

## Contact

For questions or issues, please contact the authors or open an issue on the repository.
