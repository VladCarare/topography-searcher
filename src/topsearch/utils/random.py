""" Control over the random number generation that drives a search.

    Basin hopping, the perturbations it applies, the starting vector of the
    eigenvector search and the trial rotations used for alignment all draw
    from process-wide generators. Two runs of the same script therefore
    explore different trajectories and produce different landscapes, which
    makes results impossible to reproduce exactly and comparisons between
    settings impossible to control.

    Seeding here fixes that. Note that there is more than one generator to
    set: numpy's, which scipy's random rotations also use, and the standard
    library's, which the molecular perturbations use for dihedral angles.
    Seeding only one leaves a run non-reproducible.
"""

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """ Seed every generator a search draws from.

        Parameters
        ----------
        seed : int
            Value passed to each generator. The same value gives the same
            landscape from the same starting structure and settings.
    """
    np.random.seed(seed)
    random.seed(seed)
