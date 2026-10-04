import random

import numpy as np

from topsearch.data.coordinates import StandardCoordinates
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.global_optimisation.basin_hopping import BasinHopping
from topsearch.global_optimisation.perturbations import StandardPerturbation
from topsearch.potentials.test_functions import Camelback
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.utils.random import set_global_seed


def test_seeds_both_generators():
    """ Molecular perturbations use the standard library generator while
        everything else uses numpy's, so seeding one is not enough """
    set_global_seed(7)
    first = (np.random.rand(3).tolist(), random.random())
    set_global_seed(7)
    assert (np.random.rand(3).tolist(), random.random()) == first


def test_different_seeds_differ():
    set_global_seed(1)
    first = np.random.rand(3).tolist()
    set_global_seed(2)
    assert np.random.rand(3).tolist() != first


def _run_basin_hopping(seed: int) -> list:
    """ Run a short search and return the minima it located """
    set_global_seed(seed)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    optimiser = BasinHopping(ktn=ktn, potential=Camelback(),
                             similarity=StandardSimilarity(0.05, 0.05),
                             step_taking=StandardPerturbation(max_displacement=0.7))
    optimiser.run(coords=coords, n_steps=8, conv_crit=1e-6, temperature=1.0)
    return sorted(ktn.get_minimum_energy(i) for i in range(ktn.n_minima))


def test_a_search_is_reproducible():
    """ The point of seeding: the same seed gives the same landscape """
    minima = _run_basin_hopping(42)
    # Guard against a vacuous assertion: the search has to have explored
    assert len(minima) > 1
    assert _run_basin_hopping(42) == minima


def test_the_seed_actually_matters():
    """ If the search were deterministic the test above would pass for the
        wrong reason, so check that different seeds do diverge """
    results = [tuple(_run_basin_hopping(s)) for s in (1, 2, 3, 4)]
    assert len(set(results)) > 1
