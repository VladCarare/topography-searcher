"""Shared pytest configuration."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def seed_global_rng():
    """ Seed numpy's global RNG before every test.

        Several searches draw from it directly - the starting eigenvector in
        hybrid eigenvector following, basin hopping's acceptance draw and its
        perturbations - so a test's result depends on how much randomness the
        tests before it happened to consume. That made outcomes depend on
        collection order and on whether unrelated tests were skipped. Seeding
        per test makes the suite reproducible.
    """
    np.random.seed(0)
