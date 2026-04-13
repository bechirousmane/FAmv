from abc import ABC, abstractmethod
import numpy as np

class Dimension(ABC) :
    """
    Abstract base class representing a single dimension of a search space.

    A dimension defines:
    - how values are sampled initially,
    - how candidate values are projected back into the feasible domain.

    This abstraction allows the definition of heterogeneous search spaces,
    including continuous, discrete, or custom variable types.
    """
    @abstractmethod
    def sample(self,rng:np.random.RandomState) :
        """
        Samples a feasible value for this dimension.

        Parameters
        ----------
        rng : np.random.RandomState
            Random number generator used for reproducibility.

        Returns
        -------
        object
            A sampled value belonging to the domain of this dimension.
        """
        pass

    @abstractmethod
    def project(self,value) :
        """
        Projects a value back into the feasible domain of this dimension.

        This method is typically used after an optimization step to ensure
        that constraints or domain boundaries are respected.

        Parameters
        ----------
        value : object
            Candidate value to be projected.

        Returns
        -------
        object
            Feasible value after projection.
        """
        pass

class ContinuousDimension(Dimension) :
    """
    Continuous-valued dimension defined over a bounded interval.

    The domain is defined as [lower, upper]. Sampling is uniform, and
    projection is performed via clipping.

    Parameters
    ----------
    name : str
        Name of the dimension (used for identification or debugging).

    lower : float
        Lower bound of the continuous domain.

    upper : float
        Upper bound of the continuous domain.
    """
    def __init__(self, lower, upper, name:str="") :
        
        self.name = name
        self.lower = lower
        self.upper = upper

    def sample(self, rng:np.random.RandomState):
        """
        Uniformly samples a value within the continuous bounds.

        Returns
        -------
        float
            Sampled value in [lower, upper].
        """
        return rng.uniform(self.lower, self.upper+1)
    
    def project(self, value):
        """
        Projects a value into the feasible interval using clipping.

        Parameters
        ----------
        value : float
            Candidate value.

        Returns
        -------
        float
            Value clipped to [lower, upper].
        """
        return np.clip(value, self.lower, self.upper)
    
class DiscreteDimension(Dimension) :
    """
    Discrete-valued dimension with a finite set of admissible values.

    Projection is handled by a user-defined rule to ensure flexibility,
    particularly for mixed-variable optimization problems.

    Parameters
    ----------
    name : str
        Name of the dimension.

    values : list
        List of admissible discrete values.

    projection_rules : callable, optional
        Function defining how invalid or continuous values are mapped
        to admissible discrete values.
    """
    def __init__(self,values, projection_rules:callable=None, name:str="") :
        
        self.name = name
        self.values = values
        self.projection_rules = projection_rules

    def sample(self, rng:np.random.RandomState) :
        """
        Randomly samples one value from the discrete domain.

        Returns
        -------
        object
            Sampled discrete value.
        """
        return rng.choice(self.values)
    
    def project(self, value):
        """
        Projects a value into the discrete domain using the projection rule.

        Parameters
        ----------
        value : object
            Candidate value.

        Returns
        -------
        object
            Valid discrete value after projection.
        """
        if self.projection_rules is not None :
            return self.projection_rules(value)
        
        return value