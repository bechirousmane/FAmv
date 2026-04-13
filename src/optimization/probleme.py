from search_space.search_space import SearchSpace
from search_space.particle import Particle

class Probleme :
    """
        Definition of an optimization problem.

        This class binds together:
        - the search space,
        - the fitness (objective) function,
        - the population of particles,
        - the indexing of continuous and discrete variables.

        It serves as the central interface between the optimization algorithm
        and the problem definition.

        Parameters
        ----------
        search_space : SearchSpace
            Search space defining the domains of all variables.

        continuous_idx : list of int
            Indices of continuous variables in the position vector.

        discrete_idx : list of int
            Indices of discrete variables in the position vector.

        fitness_func : callable
            Objective function to be minimized. It takes a position vector
            as input and returns a scalar fitness value.

        population_size : int, optional
            Number of particles in the population.
    """
    def __init__(
            self, 
            search_space:SearchSpace, 
            continuous_idx:list[int], 
            discrete_idx:list[int],
            fitness_func:callable,
            population_size:int=20):
        
        self.search_space = search_space
        self.continuous_idx = continuous_idx
        self.discrete_idx = discrete_idx
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.population = []

    def init_population(self) :
        """
        Initializes the particle population by sampling the search space.

        Each particle is initialized with a random feasible position.
        """
        self.population = [
            Particle(self.search_space.sample())
            for _ in range(self.population_size)
        ]
    