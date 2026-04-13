from optimization.optimization import Optimization
from optimization.probleme import Probleme
from abc import ABC, abstractmethod
import numpy as np

class FABASE(Optimization, ABC): 
    """
    Abstract base class for Firefly Algorithm (FA).

    This class implements the common mechanisms shared by all variants of the
    Firefly Algorithm, including:

    - Firefly attractiveness computation
    - Position and fitness update
    - Global and personal best tracking
    - Optional adaptive hyperparameter control

    The class is designed to be extended to support:
    - Continuous FA
    - Discrete FA
    - Mixed-variable FA (MVFA)

    Concrete subclasses must implement:
    - `move_particle`
    - `update_specific_hyperparameter`

    Parameters
        ----------
        probleme : Probleme
            Optimization problem instance containing the search space,
            population, and objective function.

        alpha : float, default=0.2
            Randomization parameter controlling the stochastic movement
            of fireflies.

        beta0 : float, default=0.5
            Attractiveness at zero distance.

        gamma : float, default=0.5
            Light absorption coefficient controlling the decay of
            attractiveness with distance.

        adaptive : bool, default=False
            Whether adaptive control of hyperparameters is enabled.

        generation : int, default=30
            Maximum number of generations.

        max_evaluations : int or None, default=None
            Maximum number of objective function evaluations.

        duration : int, default=60
            Maximum execution time in seconds.

        seed : int, default=42
            Random seed for reproducibility.

        n_processes : int or None, default=None
            Number of parallel processes for fitness evaluation.
    """

    def __init__(
            self,
            probleme:Probleme, 
            alpha:float=0.2, 
            beta0:float=0.5,
            gamma:float=0.5,
            adaptive:bool=False,
            generation:int=30, 
            max_evaluations:int=None, 
            duration:int=60, 
            seed:int=42, 
            n_processes:int=None
        ):
        super().__init__(probleme, generation, max_evaluations, duration, seed, n_processes)
        self.alpha = alpha
        self.alpha_init = alpha
        self.gamma = gamma
        self.beta0 = beta0
        self.adaptive = adaptive
        self.eval_during_updating = True

    def _attractiveness(self, r: float) -> float:
        """
        Compute the attractiveness at a given distance.

        The attractiveness decreases exponentially with the squared
        distance following the standard Firefly Algorithm formulation:

            beta(r) = beta0 * exp(-gamma * r^2)

        Parameters
        ----------
        r : float
            Distance between two fireflies.

        Returns
        -------
        float
            Attractiveness value at distance r.
        """
        return self.beta0 * np.exp(-self.gamma * r**2)
    
    
    def update_position(self, new_positions, results_eval):
        """
        Update the population with newly computed positions and fitness values.

        This method:
        - Updates each firefly's current position and fitness
        - Updates personal bests
        - Updates the global best solution
        - Stores the best fitness history

        Parameters
        ----------
        new_positions : list[list[float]]
            New positions computed by the movement strategy.

        results_eval : list[float]
            Fitness values associated with the new positions.

        Returns
        -------
        Any
            Result of the `move_particle` method (typically new candidate
            positions for the next generation).
        """
        
        return self.move_particles()
    
    
    @abstractmethod
    def distance(self, x1:list[object], x2:list[object]) -> list[object] :
        pass
    
    def move_particle_toward(self, position_i: np.ndarray, position_j: np.ndarray, r_ij:float) -> np.ndarray:
        """
        Move particle i toward particle j using FA equation.
        
        Parameters
        ----------
        position_i : np.ndarray
            Current position of particle i

        position_j : np.ndarray
            Position of brighter particle j

        r : float
            distance between two particles.
            
        Returns
        -------
        np.ndarray
            New position after movement
        """
        
        # Compute attractiveness
        beta = self._attractiveness(r_ij)
        
        # Generate random step
        n_dim = len(position_i)
        epsilon = self.rng.uniform(-0.5, 0.5, n_dim)
        
        # Movement equation: x_i = x_i + beta * (x_j - x_i) + alpha * epsilon
        new_position = position_i + beta * (position_j - position_i) + self.alpha * epsilon
        
        return new_position


    @abstractmethod
    def move_particles(self) -> tuple[list[list[object]], list[float]]:
        """
        Compute the movement of fireflies.

        This method must implement the FA movement equation and return
        the new candidate positions.

        Returns
        -------
        list[list[float]]
            New positions for the population.
        """
        pass
            
    
    def update_hyperparameters(self):
        # if not self.adaptive or self.generation < 2 :
        #     return
        # self.update_specific_hyperparameter()
        pass
        
    @abstractmethod
    def update_specific_hyperparameter(self) :
        pass

class FA(FABASE) :
    """
    Standard Firefly Algorithm for continuous optimization.

    The Firefly Algorithm is a nature-inspired metaheuristic based on the
    flashing behavior of fireflies. Each firefly is attracted to brighter
    (better fitness) fireflies, with the attraction decreasing with distance.

    Movement Equation
    -----------------
    For firefly i attracted to brighter firefly j:

        x_i(t+1) = x_i(t) + beta(r_ij) * (x_j(t) - x_i(t)) + alpha * epsilon_i

    where:
    - beta(r_ij) = beta0 * exp(-gamma * r_ij^2) is the attractiveness
    - r_ij is the Euclidean distance between fireflies i and j
    - alpha is the randomization parameter
    - epsilon_i is a random vector from a uniform or Gaussian distribution

    Parameters
    ----------
    probleme : Probleme
        Optimization problem to solve.

    alpha : float, default=0.2
        Randomization parameter (step size for random walk).
        Typical range: [0.01, 1.0]

    beta0 : float, default=1.0
        Maximum attractiveness (at r=0).
        Typical range: [0.5, 2.0]

    gamma : float, default=1.0
        Light absorption coefficient.
        Typical range: [0.01, 100]
        - Low gamma: slow decrease of attractiveness (global search)
        - High gamma: fast decrease of attractiveness (local search)

    adaptive : bool, default=False
        Enable adaptive control of alpha parameter.

    generation : int, default=30
        Maximum number of generations.

    max_evaluations : int, optional
        Maximum number of fitness evaluations.

    duration : int, default=60
        Maximum execution time in seconds.

    seed : int, default=42
        Random seed for reproducibility.

    n_processes : int, optional
        Number of parallel processes for fitness evaluation.

    Notes
    -----
    - The algorithm maintains a population of fireflies
    - Each firefly has a position in the search space and a light intensity
      (fitness value)
    - Fireflies move toward brighter fireflies
    - The brightest firefly moves randomly
    - The algorithm naturally balances exploration and exploitation

    References
    ----------
    Yang, X. S. (2008). Nature-Inspired Metaheuristic Algorithms.
    Luniver Press.
    """
    def __init__(
            self, 
            probleme: Probleme, 
            alpha: float = 0.2, 
            beta0: float = 1.0, 
            gamma: float = 1.0, 
            adaptive: bool = False, 
            generation: int = 30, 
            max_evaluations: int = None, 
            duration: int = 60, 
            seed: int = 42, 
            n_processes: int = None
        ):
        super().__init__(
            probleme, alpha, beta0, gamma, adaptive, 
            generation, max_evaluations, duration, seed, n_processes
        )

    def distance(self, x1: list[float], x2: list[float]) -> float:
        """
        Compute the Euclidean distance between two fireflies.

        Parameters
        ----------
        x1 : list[float]
            Position of the first firefly.

        x2 : list[float]
            Position of the second firefly.

        Returns
        -------
        float
            Euclidean distance between the two positions.
        """
        return np.linalg.norm(np.array(x1) - np.array(x2))
    
    def move_particles(self) -> tuple[list[list[object]], list[float]]:
        """
        Implement the standard Firefly Algorithm movement.

        For each firefly i:
        1. Compare with all other fireflies j
        2. If firefly j is brighter (better fitness), move toward it
        3. Add random walk component
        4. If no brighter firefly found, perform random walk only

        The movement equation is:
            x_i = x_i + beta(r_ij) * (x_j - x_i) + alpha * (rand - 0.5)

        Returns
        -------
        list[list[float]]
            New positions for all fireflies.
        """
        new_positions = []
        new_fitness = []
        n_dim = len(self.probleme.continuous_idx) + len(self.probleme.discrete_idx)

        for i, particle_i in enumerate(self.probleme.population):
            position_i = np.array(particle_i.position.copy())
            fitness_i = particle_i.fitness
            moved = False

            self.update_specific_hyperparameter()
            
            # Compare with all other fireflies
            for j, particle_j in enumerate(self.probleme.population):
                if i == j:
                    continue
               # print(fitness_i)
                
                # Move toward brighter (better) fireflies
                if particle_j.fitness < fitness_i:
                    position_j = np.array(particle_j.position)

                    r_ij = self.distance(position_i, position_j)
                    
                    position_i = self.move_particle_toward(position_i, position_j, r_ij)

                    position_i_proj = self.probleme.search_space.project(position_i.tolist())
                    fitness_i = self.evaluate(position_i_proj)
                    position_i = np.array(position_i_proj)

                    particle_i.position = position_i.copy()
                    particle_i.fitness = fitness_i

                    if particle_i.best_fitness > fitness_i :
                        particle_i.best_position = position_i.copy()
                        particle_i.best_fitness = fitness_i
                    
                    if self.best_fitness > fitness_i :
                        self.best_fitness = fitness_i
                        self.best_particle = particle_i

                    self.best_fitness_historie.append(self.best_fitness)
                    
                    moved = True

            
            # If no brighter firefly found, perform random walk only
            if not moved:
                epsilon = self.rng.uniform(-0.5, 0.5, n_dim)
                position_i = position_i + self.alpha * epsilon

                position_i_proj = self.probleme.search_space.project(position_i.tolist())
                fitness_i = self.evaluate(position_i_proj)
                position_i = np.array(position_i_proj)

                if particle_i.best_fitness > fitness_i :
                    particle_i.best_position = position_i.copy()
                    particle_i.best_fitness = fitness_i
                
                if self.best_fitness > fitness_i :
                    self.best_fitness = fitness_i
                    self.best_particle = particle_i
                self.best_fitness_historie.append(self.best_fitness)
            
            
            # Project to feasible space
            position_i_projected = self.probleme.search_space.project(position_i.tolist())
            new_positions.append(position_i_projected)
            new_fitness.append(fitness_i)

        return new_positions, new_fitness
    
    def update_specific_hyperparameter(self):
        """
        Adaptive control of the randomization parameter alpha.

        The alpha parameter is decreased over time to shift from exploration
        (early stages) to exploitation (later stages).

        """
        # Linear decrease
        if self.max_evaluations is not None:
            # Use evaluation-based decrease
            progress = self.evaluations / self.max_evaluations
        else:
            # Use generation-based decrease
            progress = self.curent_generation / self.generation
        
        
        self.alpha = max(0.05, self.alpha_init * (1 - progress))
    
    def __str__(self):
        return "FA"
    
class FA_Set_Distance_Based_mv(FABASE, ABC) : 
    """
    Abstract Firefly Algorithm for Mixed-Variable Optimization
    using Set-Based Distance Metrics.

    This abstract class implements the core movement and update logic
    for Firefly Algorithms (FA) operating on mixed-variable search spaces,
    i.e. problems involving both continuous and discrete variables.

    The class relies on a *set-based distance* between two candidate solutions,
    allowing discrete variables to influence attraction and movement decisions
    without requiring vector arithmetic.

    
    Parameters
    ----------
    int_val : bool, default=False
        If True: treats discrete variables as ORDINAL (adds noise to values)
        If False: treats discrete variables as CATEGORICAL (Additions of noises based on a probability)

    probleme : Probleme
        Optimization problem to solve.

    alpha : float, default=0.2
        Randomization parameter (step size for random walk).

    alpha_d : int, default=10
        Randomization parameter for integer discrete variables.

    k : float, default=1.0
        a controlling factor in the transition between exploration and exploitation

    beta0 : float, default=1.0
        Maximum attractiveness (at r=0).
        Typical range: [0.5, 2.0]

    gamma : float, default=1.0
        Light absorption coefficient.
        Typical range: [0.001, 100]
        - Low gamma: slow decrease of attractiveness (global search)
        - High gamma: fast decrease of attractiveness (local search)

    adaptive : bool, default=False
        Enable adaptive control of alpha parameter.

    generation : int, default=30
        Maximum number of generations.

    max_evaluations : int, optional
        Maximum number of fitness evaluations.

    duration : int, default=60
        Maximum execution time in seconds.

    seed : int, default=42
        Random seed for reproducibility.

    n_processes : int, optional
        Number of parallel processes for fitness evaluation.

    References
    ----------
    The discrete movement mechanisms implemented through the
    `beta_step` and `alpha_step` procedures are based on the
    Discrete Firefly Algorithm formulation used in:

    - Li, X., Gao, Y., Zhang, H., & Wang, J. (2014).
      *Multi-objective loading pattern enhancement of PWR based
      on the Discrete Firefly Algorithm*.
      Annals of Nuclear Energy, 63, 366–378.

    In this work, the authors adapt the classical Firefly Algorithm
    to discrete and combinatorial optimization problems by:
    - Replacing vector-based attraction with probabilistic
      component-wise copying (beta-step).
    - Introducing a random perturbation mechanism (alpha-step)
      to maintain diversity and avoid premature convergence.

    See Also
    --------
    FA_Hamming_mv
    FA_Gower_mv
    """
    def __init__(
            self, 
            probleme: Probleme, 
            alpha: float = 0.2, 
            alpha_d: int = 10,
            k:float = 1.0,
            beta0: float = 1.0, 
            gamma: float = 1.0, 
            int_val: float = False,
            adaptive: bool = False, 
            generation: int = 30, 
            max_evaluations: int = None, 
            duration: int = 60, 
            seed: int = 42, 
            n_processes: int = None
        ):
        super().__init__(
            probleme, alpha, beta0, gamma, adaptive, 
            generation, max_evaluations, duration, seed, n_processes
        )
        self.int_val = int_val
        self.alpha_d = alpha_d
        self.alpha_d_init = alpha_d
        self.gamma_init = gamma
        self.k = k


    
    def _discrete_attractiveness(self, r:float) -> list[object] :
        """
        Compute the attractiveness for discrete variables.

        Parameters
        ----------
        r : float
            Distance between the two solutions.

        Returns
        -------
        float
            Discrete attractiveness value in [0, 1].

        """
        return np.exp(-self.gamma*r**2)
    
    def _beta_step(self, x1:list[object], x2:list[object], r:float) -> list[object] :
        """
        Perform the beta (attraction) step for discrete variables.

        This operation mimics attraction in a discrete search space
        using set-based transitions instead of vector arithmetic.

        Parameters
        ----------
        x1 : list[object]
            Current discrete position of firefly i.

        x2 : list[object]
            Discrete position of brighter firefly j.

        r : float
            Distance between the two fireflies.

        Returns
        -------
        list[object]
            Updated discrete position after beta-step.
        """
        beta = self._discrete_attractiveness(r=r)

        result = []
        for x1_i, x2_i in zip(x1, x2) :
            rand = self.rng.rand()

            if rand < beta :
                result.append(x2_i)
            else :
                result.append(x1_i)

        return result
    
    def _alpha_step(self, x:list[object]) -> list[object] :
        """
        Apply alpha-based random perturbation to discrete variables.

        If `int_val` is False, the discrete position is returned unchanged.

        This step introduces controlled stochastic exploration
        in the discrete subspace.

        Parameters
        ----------
        x : list[object]
            Discrete position vector.

        Returns
        -------
        list[object]
            Perturbed discrete position.
        """
        if self.int_val:
            # Ordinal case
            n_dim = len(x)
            epsilon = self.rng.uniform(-2.5, 2.5, n_dim)
            result = (np.array(x) + self.alpha_d * epsilon).astype(int)
            return result.tolist()
        
        else:
            # Categorical case : probabilistic shift
            result = []
            
            for i, (dim_idx, current_value) in enumerate(zip(self.probleme.discrete_idx, x)):
                dimension = self.probleme.search_space.dimensions[dim_idx]
                values_set = dimension.values
                
                if self.rng.rand() < 1/(1 + np.exp(-self.k*(self.alpha_d - self.alpha_d_init/2))):
                    # Perturb: choose a random value from the set
                    new_value = self.rng.choice(values_set)
                    result.append(new_value)
                else:
                    # Keep current value
                    result.append(current_value)
            
            return result

    
    def _move_discrete_particle_toward(self, position1, position2, r:float) :
        """
        Move a discrete solution toward another.

        Parameters
        ----------
        position1 : list[object]
            Discrete position of firefly i.

        position2 : list[object]
            Discrete position of brighter firefly j.

        r : float
            Distance between the two fireflies.

        Returns
        -------
        list[object]
            Updated discrete position.
        """
        position_res = self._beta_step(position1, position2, r)
        position_res = self._alpha_step(position_res)
        return position_res
    
    def move_particles(self) -> tuple[list[list[object]], list[float]]:
        """
        Perform one iteration of the mixed-variable Firefly Algorithm. 

        Returns
        -------
        tuple[list[list[object]], list[float]]
            - New positions of all fireflies
            - Corresponding fitness values

        """ 
        new_positions = []
        new_fitness = []

        self.update_specific_hyperparameter()

        for i, particle_i in enumerate(self.probleme.population):
            position_i = particle_i.position.copy()
            
            # Separate continuous and discrete parts
            continuous_pos_i = np.array([position_i[dim] for dim in self.probleme.continuous_idx], dtype=float)
            discrete_pos_i = [position_i[dim] for dim in self.probleme.discrete_idx]
            
            fitness_i = particle_i.fitness
            moved = False
            
            # Compare with all other fireflies
            for j, particle_j in enumerate(self.probleme.population):
                if i == j:
                    continue
                
                if particle_j.fitness < fitness_i:
                    position_j = particle_j.position
                    r_ij = self.distance(position_i, position_j)
                    
                    # Separate continuous and discrete parts of j
                    continuous_pos_j = np.array([position_j[dim] for dim in self.probleme.continuous_idx], dtype=float)
                    discrete_pos_j = [position_j[dim] for dim in self.probleme.discrete_idx]
                    
                    # Move continuous part
                    continuous_pos_i = self.move_particle_toward(continuous_pos_i, continuous_pos_j, r_ij=r_ij)
                    
                    # Move discrete part
                    discrete_pos_i = self._move_discrete_particle_toward(discrete_pos_i, discrete_pos_j, r=r_ij)
                    
                    # Reconstruct full position
                    full_position = [None] * len(position_i)
                    for idx, cont_val in zip(self.probleme.continuous_idx, continuous_pos_i):
                        full_position[idx] = cont_val
                    for idx, disc_val in zip(self.probleme.discrete_idx, discrete_pos_i):
                        full_position[idx] = disc_val
                    
                    # Project and evaluate
                    full_position_proj = self.probleme.search_space.project(full_position)
                    fitness_new = self.evaluate(full_position_proj)
                    
                    particle_i.position = full_position_proj
                    particle_i.fitness = fitness_new
                    fitness_i = fitness_new
                    
                    # Update personal best if improved
                    if particle_i.best_fitness is None or fitness_new < particle_i.best_fitness:
                        particle_i.best_position = full_position_proj.copy()
                        particle_i.best_fitness = fitness_new
                    
                    # Update global best if improved
                    if self.best_fitness is None or fitness_new < self.best_fitness:
                        self.best_fitness = fitness_new
                        self.best_particle = particle_i
                    
                    # Record in history after EACH evaluation
                    self.best_fitness_historie.append(self.best_fitness)
                    
                    # Update for next iteration within this loop
                    continuous_pos_i = np.array([full_position_proj[dim] for dim in self.probleme.continuous_idx], dtype=float)
                    discrete_pos_i = [full_position_proj[dim] for dim in self.probleme.discrete_idx]
                    
                    moved = True
            
            # If no brighter firefly, perform random walk
            if not moved:
                # Random walk for continuous
                n_cont = len(continuous_pos_i)
                epsilon_cont = self.rng.uniform(-0.5, 0.5, n_cont)
                continuous_pos_i = continuous_pos_i + self.alpha * epsilon_cont
                
                # Random walk for discrete (if int_val=True)
                discrete_pos_i = self._alpha_step(discrete_pos_i)
                
                # Reconstruct and evaluate
                full_position = [None] * len(position_i)
                for idx, cont_val in zip(self.probleme.continuous_idx, continuous_pos_i):
                    full_position[idx] = cont_val
                for idx, disc_val in zip(self.probleme.discrete_idx, discrete_pos_i):
                    full_position[idx] = disc_val
                
                full_position_proj = self.probleme.search_space.project(full_position)
                fitness_new = self.evaluate(full_position_proj)
                
                # Update even after random walk
                particle_i.position = full_position_proj
                particle_i.fitness = fitness_new
                fitness_i = fitness_new
                
                # Update personal best if improved
                if particle_i.best_fitness is None or fitness_new < particle_i.best_fitness:
                    particle_i.best_position = full_position_proj.copy()
                    particle_i.best_fitness = fitness_new
                
                # Update global best if improved
                if self.best_fitness is None or fitness_new < self.best_fitness:
                    self.best_fitness = fitness_new
                    self.best_particle = particle_i
                
                # Record in history
                self.best_fitness_historie.append(self.best_fitness)
            
            # store final position and fitness for return
            new_positions.append(particle_i.position)
            new_fitness.append(particle_i.fitness)

        return new_positions, new_fitness
    
    def update_specific_hyperparameter(self):
        pass
    
    def __str__(self):
        return "FA_Hamming_mv"
    
class FA_Hamming_mv(FA_Set_Distance_Based_mv) :
    """
    Firefly Algorithm for Mixed-Variable Optimization using Hamming-based distance.

    This variant of the Firefly Algorithm (FA) is designed for optimization problems
    involving both continuous and discrete decision variables.

    The distance between two fireflies is computed as a combination of:
    - Euclidean distance on continuous variables
    - Hamming distance on discrete variables

    Continuous variables are updated using the standard FA movement equation,
    while discrete variables are updated using a probabilistic, set-based rule
    inspired by Hamming distance.

    Distance Definition
    -------------------
    Let x and y be two solutions:

        r(x, y) = ||x_c - y_c||_2 + H(x_d, y_d)

    where:
    - x_c, y_c are continuous components
    - x_d, y_d are discrete components
    - H(·) is the Hamming distance
    
    
    Parameters
    ----------
    int_val : bool, default=False
        If True: treats discrete variables as ORDINAL (adds noise to values)
        If False: treats discrete variables as CATEGORICAL (Additions of noises based on a probability)

    probleme : Probleme
        Optimization problem to solve.

    alpha : float, default=0.2
        Randomization parameter (step size for random walk).

    alpha_d : int, default=10
        Randomization parameter for integer discrete variables.

    k : float, default=1.0
        a controlling factor in the transition between exploration and exploitation

    beta0 : float, default=1.0
        Maximum attractiveness (at r=0).
        Typical range: [0.5, 2.0]

    gamma : float, default=1.0
        Light absorption coefficient.
        Typical range: [0.001, 100]
        - Low gamma: slow decrease of attractiveness (global search)
        - High gamma: fast decrease of attractiveness (local search)

    adaptive : bool, default=False
        Enable adaptive control of alpha parameter.

    generation : int, default=30
        Maximum number of generations.

    max_evaluations : int, optional
        Maximum number of fitness evaluations.

    duration : int, default=60
        Maximum execution time in seconds.

    seed : int, default=42
        Random seed for reproducibility.

    n_processes : int, optional
        Number of parallel processes for fitness evaluation.

    See Also
    --------
    FA_Set_Distance_Based_mv
    FA_Gower_mv
    """
    def __init__(
            self, 
            probleme: Probleme, 
            alpha: float = 0.2,
            alpha_d: int = 10, 
            k: float = 1.0,
            beta0: float = 1.0, 
            gamma: float = 1.0, 
            int_val: float = False,
            adaptive: bool = False, 
            generation: int = 30, 
            max_evaluations: int = None, 
            duration: int = 60, 
            seed: int = 42, 
            n_processes: int = None
        ):
        super().__init__(
            probleme, alpha, alpha_d, k, beta0, gamma, int_val, adaptive, 
            generation, max_evaluations, duration, seed, n_processes
        )


    def _hamming_distance(self, x1:list[object], x2:list[object]) -> int :
        hamming_distance = 0
        for elem_x1, elem_x2 in zip(x1,x2) :
            if elem_x1 != elem_x2 :
                hamming_distance += 1

        return hamming_distance
    
    def distance(self, x1, x2):
        x1_c = []
        x2_c = []
        x1_d = []
        x2_d = []

        for i in self.probleme.continuous_idx :
            x1_c.append(x1[i])
            x2_c.append(x2[i])

        for i in self.probleme.discrete_idx :
            x1_d.append(x1[i])
            x2_d.append(x2[i])

        euclid_dist = np.linalg.norm(np.array(x1_c) - np.array(x2_c))
        hamming_dist = self._hamming_distance(x1_d, x2_d)
        r = euclid_dist + hamming_dist

        return r/len(x1)
    
    def __str__(self):
        return "FA_Hamming_mv"
    
class FA_Hamming_mv_adaptive_alpha(FA_Hamming_mv) :
    def __init__(
            self, 
            probleme: Probleme, 
            alpha: float = 0.2,
            alpha_d: int = 10, 
            beta0: float = 1.0, 
            gamma: float = 1.0, 
            int_val: float = False,
            adaptive: bool = True, 
            generation: int = 30, 
            max_evaluations: int = None, 
            duration: int = 60, 
            seed: int = 42, 
            n_processes: int = None
        ):
        super().__init__(
            probleme, alpha, alpha_d, beta0, gamma, int_val, adaptive, 
            generation, max_evaluations, duration, seed, n_processes
        )

    def update_specific_hyperparameter(self):
        """Linear decrease of alpha parameter"""
        if self.max_evaluations is not None:
            progress = self.evaluations / self.max_evaluations
        else:
            progress = self.curent_generation / self.generation
        
        self.alpha = max(0.01, self.alpha_init * (1 - progress))
        self.alpha_d = max(1, self.alpha_d_init * (1 - progress))

class FA_Hamming_mv_adaptive_gamma(FA_Hamming_mv) :
    def __init__(
            self, 
            probleme: Probleme, 
            alpha: float = 0.2,
            alpha_d: int = 10, 
            beta0: float = 1.0, 
            gamma: float = 1.0, 
            int_val: float = False,
            adaptive: bool = True, 
            generation: int = 30, 
            max_evaluations: int = None, 
            duration: int = 60, 
            seed: int = 42, 
            n_processes: int = None
        ):
        super().__init__(
            probleme, alpha, alpha_d, beta0, gamma, int_val, adaptive, 
            generation, max_evaluations, duration, seed, n_processes
        )

    def update_specific_hyperparameter(self):
        """Linear decrease of gamma parameter"""
        if self.max_evaluations is not None:
            progress = self.evaluations / self.max_evaluations
        else:
            progress = self.curent_generation / self.generation
        
        self.gamma = max(0.01, self.gamma_init * (1 - progress))    

class FA_Hamming_mv_adaptive_alpha_gamma(FA_Hamming_mv) :
    def __init__(
            self, 
            probleme: Probleme, 
            alpha: float = 0.2,
            alpha_d: int = 10, 
            beta0: float = 1.0, 
            gamma: float = 1.0, 
            int_val: float = False,
            adaptive: bool = True, 
            generation: int = 30, 
            max_evaluations: int = None, 
            duration: int = 60, 
            seed: int = 42, 
            n_processes: int = None
        ):
        super().__init__(
            probleme, alpha, alpha_d, beta0, gamma, int_val, adaptive, 
            generation, max_evaluations, duration, seed, n_processes
        )

    def update_specific_hyperparameter(self):
        """Linear decrease of both alpha and gamma parameters"""
        if self.max_evaluations is not None:
            progress = self.evaluations / self.max_evaluations
        else:
            progress = self.curent_generation / self.generation
        
        self.alpha = max(0.01, self.alpha_init * (1 - progress))
        self.alpha_d = max(1, self.alpha_d_init * (1 - progress))
        self.gamma = max(0.01, self.gamma_init * (1 - progress))
    

class FA_Gower_mv(FA_Set_Distance_Based_mv) :
    """
    Firefly Algorithm for Mixed-Variable Optimization using Gower's distance.

    This variant of the Firefly Algorithm (FA) uses Gower's distance to measure
    similarity between fireflies in mixed-variable search spaces.

    Gower's distance provides a normalized and balanced measure across:
    - Continuous variables (scaled by their valid ranges)
    - Discrete variables (binary mismatch)

    This makes the algorithm particularly robust when:
    - Variables have heterogeneous scales
    - Continuous dimensions have very different ranges
    - Discrete and continuous variables must contribute equally

    Distance Definition
    -------------------
    For two solutions x and y with p total dimensions:

        r(x, y) = (1 / p) * sum d_k(x_k, y_k)

    where:
    - For continuous variable k:
          d_k = |x_k - y_k| / (upper_k - lower_k)
    - For discrete variable k:
          d_k = 0 if x_k == y_k else 1

    The resulting distance is normalized to [0, 1].

    Discrete and Continuous Movement
    --------------------------------
    - Continuous variables:
        Updated using the standard FA movement equation.
    - Discrete variables:
        Updated using a probabilistic beta-step based on Gower distance,
        followed by an optional alpha-based perturbation.

    
    
    Parameters
    ----------
    int_val : bool, default=False
        If True: treats discrete variables as ORDINAL (adds noise to values)
        If False: treats discrete variables as CATEGORICAL (Additions of noises based on a probability)

    probleme : Probleme
        Optimization problem to solve.

    alpha : float, default=0.2
        Randomization parameter (step size for random walk).

    alpha_d : int, default=10
        Randomization parameter for integer discrete variables.

    k : float, default=1.0
        a controlling factor in the transition between exploration and exploitation

    beta0 : float, default=1.0
        Maximum attractiveness (at r=0).
        Typical range: [0.5, 2.0]

    gamma : float, default=1.0
        Light absorption coefficient.
        Typical range: [0.001, 100]
        - Low gamma: slow decrease of attractiveness (global search)
        - High gamma: fast decrease of attractiveness (local search)

    adaptive : bool, default=False
        Enable adaptive control of alpha parameter.

    generation : int, default=30
        Maximum number of generations.

    max_evaluations : int, optional
        Maximum number of fitness evaluations.

    duration : int, default=60
        Maximum execution time in seconds.

    seed : int, default=42
        Random seed for reproducibility.

    n_processes : int, optional
        Number of parallel processes for fitness evaluation.

    References
    ----------
    Gower's distance was introduced in:

    - Gower, J. C. (1971).
      *A general coefficient of similarity and some of its properties*.
      Biometrics, 27(4), 857–871.

    This distance metric is specifically designed for mixed-variable
    data and provides a normalized measure of dissimilarity across
    continuous, ordinal, and categorical variables.

    See Also
    --------
    FA_Set_Distance_Based_mv
    FA_Hamming_mv
    """
    def __init__(
            self, 
            probleme: Probleme, 
            alpha: float = 0.2,
            alpha_d: int = 10, 
            k: float = 1.0,
            beta0: float = 1.0, 
            gamma: float = 1.0, 
            int_val: float = False,
            adaptive: bool = False, 
            generation: int = 30, 
            max_evaluations: int = None, 
            duration: int = 60, 
            seed: int = 42, 
            n_processes: int = None
        ):
        super().__init__(
            probleme, alpha, alpha_d, k, beta0, gamma, int_val, adaptive, 
            generation, max_evaluations, duration, seed, n_processes
        )

        self.continuous_ranges = []
        for idx in self.probleme.continuous_idx:
            dim = self.probleme.search_space.dimensions[idx]
            range_k = dim.upper - dim.lower
            self.continuous_ranges.append(range_k if range_k > 0 else 1.0)
    
    def distance(self, x1: list[object], x2: list[object]) -> float:
        """
        Gower's distance: normalized distance for mixed variables.
        """
        total_distance = 0.0
        n_dims = len(x1)
        
        # Continuous part (normalized by range)
        for i, (idx, range_k) in enumerate(zip(self.probleme.continuous_idx, self.continuous_ranges)):
            diff = abs(x1[idx] - x2[idx])
            total_distance += diff / range_k
        
        # Discrete part (0 if equal, 1 if different)
        for idx in self.probleme.discrete_idx:
            total_distance += 0 if x1[idx] == x2[idx] else 1
        
        # Average over all dimensions
        return total_distance / n_dims
    
    def __str__(self):
        return "FA_Gower_mv"
    
class FA_Gower_mv_adaptive_alpha(FA_Gower_mv) :
    def __init__(
            self, 
            probleme: Probleme, 
            alpha: float = 0.2,
            alpha_d: int = 10, 
            beta0: float = 1.0, 
            gamma: float = 1.0, 
            int_val: float = False,
            adaptive: bool = True, 
            generation: int = 30, 
            max_evaluations: int = None, 
            duration: int = 60, 
            seed: int = 42, 
            n_processes: int = None
        ):
        super().__init__(
            probleme, alpha, alpha_d, beta0, gamma, int_val, adaptive, 
            generation, max_evaluations, duration, seed, n_processes
        )

    def update_specific_hyperparameter(self):
        """Linear decrease of alpha parameter"""
        if self.max_evaluations is not None:
            progress = self.evaluations / self.max_evaluations
        else:
            progress = self.curent_generation / self.generation
        
        self.alpha = max(0.01, self.alpha_init * (1 - progress))
        self.alpha_d = max(1, self.alpha_d_init * (1 - progress))

class FA_Gower_mv_adaptive_gamma(FA_Gower_mv) :
    def __init__(
            self, 
            probleme: Probleme, 
            alpha: float = 0.2,
            alpha_d: int = 10, 
            beta0: float = 1.0, 
            gamma: float = 1.0, 
            int_val: float = False,
            adaptive: bool = True, 
            generation: int = 30, 
            max_evaluations: int = None, 
            duration: int = 60, 
            seed: int = 42, 
            n_processes: int = None
        ):
        super().__init__(
            probleme, alpha, alpha_d, beta0, gamma, int_val, adaptive, 
            generation, max_evaluations, duration, seed, n_processes
        )

    def update_specific_hyperparameter(self):
        """Linear decrease of gamma parameter"""
        if self.max_evaluations is not None:
            progress = self.evaluations / self.max_evaluations
        else:
            progress = self.curent_generation / self.generation
        
        self.gamma = max(0.01, self.gamma_init * (1 - progress))

class FA_Gower_mv_adaptive_alpha_gamma(FA_Gower_mv) :
    def __init__(
            self, 
            probleme: Probleme, 
            alpha: float = 0.2,
            alpha_d: int = 10, 
            beta0: float = 1.0, 
            gamma: float = 1.0, 
            int_val: float = False,
            adaptive: bool = True, 
            generation: int = 30, 
            max_evaluations: int = None, 
            duration: int = 60, 
            seed: int = 42, 
            n_processes: int = None
        ):
        super().__init__(
            probleme, alpha, alpha_d, beta0, gamma, int_val, adaptive, 
            generation, max_evaluations, duration, seed, n_processes
        )

    def update_specific_hyperparameter(self):
        """Linear decrease of both alpha and gamma parameters"""
        if self.max_evaluations is not None:
            progress = self.evaluations / self.max_evaluations
        else:
            progress = self.curent_generation / self.generation
        
        self.alpha = max(0.01, self.alpha_init * (1 - progress))
        self.alpha_d = max(1, self.alpha_d_init * (1 - progress))
        self.gamma = max(0.01, self.gamma_init * (1 - progress))

class FA_Gower_mv_discrete_attractiveness_func2(FA_Gower_mv_adaptive_gamma) :
    def __init__(
            self, 
            probleme: Probleme, 
            alpha: float = 0.2,
            alpha_d: int = 10, 
            beta0: float = 1.0, 
            gamma: float = 1.0, 
            int_val: float = False,
            adaptive: bool = True, 
            generation: int = 30, 
            max_evaluations: int = None, 
            duration: int = 60, 
            seed: int = 42, 
            n_processes: int = None
        ):
        super().__init__(
            probleme, alpha, alpha_d, beta0, gamma, int_val, adaptive, 
            generation, max_evaluations, duration, seed, n_processes
        )

    def _discrete_attractiveness(self, r:float) -> list[object] :
        """
        Compute the attractiveness for discrete variables using a modified function.

        Parameters
        ----------
        r : float
            Distance between the two solutions.

        Returns
        -------
        float
            Discrete attractiveness value in [0, 1].

        """
        return np.exp(-self.gamma)*(1 - r)


    






