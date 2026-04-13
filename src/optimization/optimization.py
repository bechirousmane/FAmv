from optimization.probleme import Probleme
from abc import ABC, abstractmethod
from tqdm import tqdm
import time
import numpy as np
from multiprocessing import Pool
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt


class Optimization(ABC):
    """
        Abstract base class for optimization algorithms.
        This class provides the framework for implementing various optimization
        algorithms that operate on a defined optimization problem.
        Parameters
        ----------
        probleme : Probleme
            The optimization problem to be solved.
        generation : int, optional, default=30
            Maximum number of generations (iterations) to perform. This parameter is used
            only if max_evaluations is not set.
        max_evaluations : int, optional
            Maximum number of fitness evaluations to perform.
        duration : int, optional, default=60
            Maximum duration (in seconds) for the optimization process.
        seed : int, optional, default=42
            Random seed for reproducibility.
        n_processes : int, optional
            Number of parallel processes to use for fitness evaluations.
    """
    def __init__(
            self, 
            probleme: Probleme, 
            generation: int = 30,
            max_evaluations: int = None, 
            duration: int = 60,
            seed: int = 42,
            n_processes: int = None 
    ):
        self.probleme = probleme
        self.generation = generation
        self.max_evaluations = max_evaluations
        self.duration = duration
        self.curent_generation = 1
        self.evaluations = 0
        self.best_fitness_historie = []
        self.pop_fitness_historie = []
        self.best_particle = None
        self.best_fitness = None
        self.rng = np.random.RandomState(seed)
        
        if n_processes is None:
            n_processes = multiprocessing.cpu_count()
        self.n_processes = n_processes
        self.pool = None  

        self.eval_during_updating = False
    
    def evaluate(self, position:list[object]) -> list[object] :

        result = self.probleme.fitness_func(position)

        self.evaluations += 1

        return result
    
    def evaluates(self, positions:list[list[object]]) -> list[float]:
        """
        Evaluates a list of candidate solutions.

        Fitness evaluations can be performed either sequentially or in parallel,
        depending on whether a multiprocessing pool is available.

        Parameters
        ----------
        positions : list of position vectors
            Each position represents a candidate solution in the search space.

        Returns
        -------
        list of float
            Fitness values corresponding to each position.
        """
        if self.pool is not None:
            results = self.pool.map(self.probleme.fitness_func, positions)
        else:
            results = [self.probleme.fitness_func(pos) for pos in positions]
        self.evaluations += len(positions)
        return results
    
    def _initialize_population(self):
        """
            Initializes and evaluates the initial population.

            For each particle:
            - the initial fitness is computed,
            - the personal best position is initialized,
            - the global best solution is updated accordingly.

            Returns
            -------
            list of position vectors
                Initial population positions.
        """
        new_positions = [pop.position for pop in self.probleme.population]
        
        initial_results = self.evaluates(new_positions)
        
        for particle, fitness in zip(self.probleme.population, initial_results):
            particle.fitness = fitness
            particle.best_fitness = fitness
            particle.best_position = particle.position.copy()
            
            if self.best_fitness is None or fitness < self.best_fitness:
                self.best_particle = particle
                self.best_fitness = fitness
        
        self.best_fitness_historie.append(self.best_fitness)
        self.pop_fitness_historie.append(initial_results)
        
        return new_positions

    @abstractmethod
    def update_position(self, new_positions, results_eval: list[float]) -> list[list[object]]:
        """
        Updates particle states based on evaluated fitness values.

        For each particle:
        - the current position is updated,
        - the personal best position and fitness are updated if an improvement is found,
        - the global best solution is updated if a better fitness value is encountered.

        Parameters
        ----------
        new_positions : list of position vectors
            Newly generated positions for each particle.

        results_eval : list of float
            Fitness values associated with the new positions.

        Returns
        -------
        list of position vectors
            Positions to be used in the next iteration.
        """
        pass

    @abstractmethod
    def update_hyperparameters(self):
        """
        Updates algorithm-specific hyperparameters.
        """
        pass

    def run(self, verbose: bool = False):
        """
        Executes the optimization process.

        The algorithm iteratively evaluates the population and updates particle
        positions until one of the following stopping criteria is met:
        - the maximum number of generations is reached,
        - the maximum execution time is exceeded.
        - the maximum number of fitness evaluations is reached.

        Parameters
        ----------
        verbose : bool, optional
            If True, displays convergence information and progress indicators.
        """
        self.probleme.init_population()
        
        with Pool(processes=self.n_processes) as pool:
            self.pool = None
            
            new_positions = self._initialize_population()
            
            begin = time.time()

            desc = "Evaluation" if self.max_evaluations is not None else "Generation"
            total = self.max_evaluations if self.max_evaluations is not None else self.generation

            # Stopping criteria
            def stopping_criteria():
                
                if self.max_evaluations is not None:
                    if self.evaluations >= self.max_evaluations:
                        return True
                
                if self.max_evaluations is None : 
                    if self.curent_generation > self.generation:
                        return True
                
                if (time.time() - begin) >= self.duration:
                    return True
                return False
            results = [pop.fitness for pop in self.probleme.population]
            
            with tqdm(total=total, desc=desc, disable=not verbose) as pbar:
                while not stopping_criteria():
                    # Update hyperparameters
                    self.update_hyperparameters()

                    if not self.eval_during_updating :
                        # update position
                        new_positions = self.update_position(new_positions, results_eval=results)

                        # Evaluate new positions
                        results = self.evaluates(new_positions)


                    else :
                        # update position
                        new_positions, results = self.update_position(new_positions, results_eval=results)
                    
                    
                    self.pop_fitness_historie.append(results)
                    
                    self.curent_generation += 1


                    if self.max_evaluations is not None:
                        pbar.update(self.evaluations - pbar.n)
                    else:
                        pbar.update(1)
                    
                    if verbose and self.curent_generation % 10 == 0:
                        pbar.set_postfix({
                            'best_fitness': f'{self.best_fitness:.6f}',
                            'time': f'{time.time() - begin:.2f}s'
                        })
            
        
        if verbose:
            print(f"\nOptimization completed:")
            print(f"  - Generations: {self.curent_generation - 1}")
            print(f"  - Time: {time.time() - begin:.2f}s")
            print(f"  - Best fitness: {self.best_fitness:.6f}")
    
    def get_best_solution(self):
        """
        Returns the best solution found
        @return dict : position vector, fitness and generation
        """
        if self.best_particle is None:
            return None
        return {
            'position': self.best_particle.best_position,
            'fitness': self.best_fitness,
            'generation': self.curent_generation - 1,
            'evaluations': self.evaluations
        }
    
    
    def plot_convergence_curve(self, marker="s"):
        """
        Plots the convergence curve of the best fitness over evaluation of fitness function.
        """
        sns.set_theme(style="whitegrid", context="talk")

        best_fitness = np.array(self.best_fitness_historie)

        plt.plot(np.arange(len(best_fitness)), best_fitness, label=self)

        plt.xlabel("Fitness evaluations")
        plt.ylabel("Best fitness")
        plt.title("Convergence curve (best-so-far)")
        plt.legend()
        plt.tight_layout()
        plt.show()
