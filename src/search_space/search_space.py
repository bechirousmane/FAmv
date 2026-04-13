from search_space.dimension import Dimension
import numpy as np

class SearchSpace:
    """
        Representation of a multidimensional search space.

        The search space is defined as an ordered list of Dimension objects,
        allowing the combination of continuous and discrete variables.

        Parameters
        ----------
        dimensions : list of Dimension
            List of dimensions defining the search space.

        seed : int
            Random seed used for reproducible sampling.
    """
    def __init__(self, dimensions:list[Dimension], seed:int):
        self.dimensions = dimensions
        self.rng = np.random.RandomState(seed)

    def sample(self) -> list[object] :
        """
        Samples a full position vector from the search space.

        Returns
        -------
        list of object
            Sampled position vector, one value per dimension.
        """
        return [dim.sample(self.rng) for dim in self.dimensions]
    
    def project(self, positions) :
        """
        Projects a position vector into the feasible search space.

        Parameters
        ----------
        positions : list
            Candidate position vector.

        Returns
        -------
        list
            Feasible position vector after projection.
        """
        return [
            dim.project(position) for dim, position in zip(self.dimensions, positions)
        ]