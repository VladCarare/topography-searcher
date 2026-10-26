"""Two minima related by a symmetry of the molecule: one structure, or two?

closest_distance searches over atom permutations, so a structure and its
automorphic image come back at essentially zero distance and the search merges
them. That is right when the two are the same point found twice, and wrong when
they are the two ends of a degenerate rearrangement: merging them turns the
saddle between them into a self-loop, and closest_enumeration cannot propose a
minimum against itself, so the rearrangement is never searched for.

keep_symmetry_copies makes the merge test ask whether the structures match
WITHOUT relabelling, which separates the two cases.
"""

import os

import ase.io
import numpy as np
import pytest

from topsearch.data.coordinates import MolecularCoordinates
from topsearch.similarity.molecular_similarity import MolecularSimilarity

current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

ENERGY = -1.0


def azobenzene():
    """Azobenzene and a non-trivial automorphism of its bond graph.

    Azobenzene is used because its phenyl rings make the automorphism an exact
    symmetry rather than an approximate one, so the permuted distance is zero
    to numerical precision and the test does not depend on a tolerance.
    """
    atoms = ase.io.read(f"{current_dir}/test_data/azobenzene.xyz")
    coords = MolecularCoordinates(
        atoms.get_chemical_symbols(), atoms.get_positions().flatten()
    )
    identity = list(range(len(atoms)))
    # Pick a non-identity automorphism explicitly: isomorphisms_iter yields in
    # no guaranteed order, so indexing into the list would be a latent bug.
    permutations = [a for a in coords.automorphisms if a != identity]
    if not permutations:
        pytest.skip("no non-trivial automorphism found for azobenzene")
    image = coords.position.reshape(-1, 3)[permutations[0]].flatten()
    return coords, image


def comparer(keep_symmetry_copies):
    return MolecularSimilarity(
        distance_criterion=0.3,
        energy_criterion=5e-3,
        weighted=False,
        allow_inversion=True,
        keep_symmetry_copies=keep_symmetry_copies,
    )


def test_symmetry_copy_merged_by_default():
    """Without the flag, an automorphic image is the same minimum."""
    coords, image = azobenzene()
    assert comparer(False).test_same(coords, image, ENERGY, ENERGY)


def test_symmetry_copy_kept_apart_with_flag():
    """With the flag, it is a second copy and is kept."""
    coords, image = azobenzene()
    assert not comparer(True).test_same(coords, image, ENERGY, ENERGY)


@pytest.mark.parametrize("keep", [False, True])
def test_identical_structures_always_merge(keep):
    """The flag must not stop a genuine repeat being recognised.

    This is the half that matters: a change making the flag reject everything
    would still pass the two tests above.
    """
    coords, _ = azobenzene()
    assert comparer(keep).test_same(
        coords, coords.position.copy(), ENERGY, ENERGY
    )


def test_permuted_and_in_place_distances_differ():
    """The two distances are what the flag distinguishes between."""
    coords, image = azobenzene()
    same = comparer(True)
    assert same.closest_distance(coords, image.copy()) < 1e-6
    assert same.in_place_distance(coords, image.copy()) > 1.0


def test_in_place_distance_leaves_its_argument_alone():
    """in_place_distance centres coords1 in place and must restore it.

    It writes to coords1.position because rotational_alignment reads the
    position off the object rather than taking it as an argument. The
    restoration is in a finally block; this catches a refactor that drops it.
    """
    coords, image = azobenzene()
    before = coords.position.copy()
    comparer(True).in_place_distance(coords, image)
    assert np.allclose(coords.position, before)
