import numpy as np

class Particle :
    """
    Representation of a particle in population-based optimization algorithms.

    A particle stores:
    - its current position,
    - its current fitness value,
    - its personal best position,
    - its personal best fitness.

    Parameters
    ----------
    position : list
        Initial position vector of the particle.
    """
    def __init__(self, position):
        self.position = position
        self.fitness = None
        self.best_position = None
        self.best_fitness = None
