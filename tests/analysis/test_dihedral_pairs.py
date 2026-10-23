""" Tests for selecting connection pairs by how many dihedrals separate them.

    closest_enumeration ranks neighbours by RMSD, which only loosely tracks
    the number of rotations an interpolation has to perform. Measured on
    aspirin under GFN2-xTB, 45% of the pairs it picks differ by three
    dihedrals, and those carry a 16% chance of producing an off-reference
    structure against 0% for one and 2% for two. Ranking by dihedral
    separation instead keeps the interpolations short, and the landscape
    stays connected because the remaining pairs reach each other through
    intermediate minima.

    These tests cover the mechanics only. The landscape comparison was run
    separately and came out flat, so the selector is kept as a recorded
    negative result rather than as the recommended choice.
"""

import os

import numpy as np
import pytest
from ase.io import read

from topsearch.analysis.pair_selection import (dihedral_enumeration,
                                               dihedral_separation,
                                               rotatable_dihedrals)
from topsearch.data.coordinates import MolecularCoordinates
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork

current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TEMPLATES = f"{current_dir}/test_data"


def network_of(molecule, source):
    """A network holding the minima from a finished landscape run."""
    species = read(f"{TEMPLATES}/{molecule}_relax.xyz").get_chemical_symbols()
    ktn = KineticTransitionNetwork()
    for n, frame in enumerate(read(source, ":")):
        ktn.add_minimum(frame.get_positions().flatten(), float(n))
    return ktn, MolecularCoordinates(species, read(source, ":")[0]
                                     .get_positions().flatten())


def test_rotatable_dihedrals_finds_the_right_count():
    """One dihedral per bond that has substituents at both ends."""
    species = read(f"{TEMPLATES}/malonaldehyde_relax.xyz")
    coords = MolecularCoordinates(species.get_chemical_symbols(),
                                  species.get_positions().flatten())
    assert len(rotatable_dihedrals(coords)) == 2


def test_separation_is_zero_for_a_structure_against_itself():
    species = read(f"{TEMPLATES}/salicylic_relax.xyz")
    coords = MolecularCoordinates(species.get_chemical_symbols(),
                                  species.get_positions().flatten())
    dihedrals = rotatable_dihedrals(coords)
    position = coords.position
    assert dihedral_separation(coords, position, position, dihedrals) == 0


def test_separation_counts_the_angles_the_interpolation_must_turn():
    """Turning one bond can change several dihedral angles, and should count
    as several.

    Dihedral angles are not independent: rotating about one bond moves atoms
    that other dihedrals are measured from, and in a ring the coupling is
    strong. Turning the salicylic acid bond (8, 0, 1, 2) by 120 degrees moves
    four other angles by 26 to 124 degrees.

    That is the right thing to count here. The interpolation works in the
    space of these angles, so a pair whose minima differ in five of them asks
    the interpolation to turn five at once, however few bonds a chemist would
    say had rotated. The measure is of interpolation difficulty, not of
    mechanism.
    """
    species = read(f"{TEMPLATES}/salicylic_relax.xyz")
    coords = MolecularCoordinates(species.get_chemical_symbols(),
                                  species.get_positions().flatten())
    dihedrals = rotatable_dihedrals(coords)
    turned = species.copy()
    turned.set_dihedral(*dihedrals[0],
                        turned.get_dihedral(*dihedrals[0]) + 120.0)
    separation = dihedral_separation(coords, coords.position,
                                     turned.get_positions().flatten(),
                                     dihedrals)
    assert separation >= 1
    assert separation <= len(dihedrals)


def test_separation_grows_with_the_number_of_bonds_turned():
    """Turning more bonds cannot reduce the count."""
    species = read(f"{TEMPLATES}/salicylic_relax.xyz")
    coords = MolecularCoordinates(species.get_chemical_symbols(),
                                  species.get_positions().flatten())
    dihedrals = rotatable_dihedrals(coords)
    one, two = species.copy(), species.copy()
    one.set_dihedral(*dihedrals[0], one.get_dihedral(*dihedrals[0]) + 120.0)
    two.set_dihedral(*dihedrals[0], two.get_dihedral(*dihedrals[0]) + 120.0)
    two.set_dihedral(*dihedrals[-1], two.get_dihedral(*dihedrals[-1]) + 120.0)
    first = dihedral_separation(coords, coords.position,
                                one.get_positions().flatten(), dihedrals)
    second = dihedral_separation(coords, coords.position,
                                 two.get_positions().flatten(), dihedrals)
    assert second >= first


def test_enumeration_prefers_short_moves():
    """Every pair chosen is within the threshold that keeps minima reachable."""
    source = f"{TEMPLATES}/salicylic_minima.xyz"
    ktn, coords = network_of("salicylic", source)
    pairs, limit = dihedral_enumeration(ktn, coords, neighbours=2,
                                        report_limit=True)
    dihedrals = rotatable_dihedrals(coords)
    for i, j in pairs:
        assert dihedral_separation(coords, ktn.get_minimum_coords(i),
                                   ktn.get_minimum_coords(j),
                                   dihedrals) <= limit


def test_enumeration_leaves_no_minimum_stranded():
    """The pairs must still reach every minimum, or connectivity is lost."""
    import networkx as nx
    for molecule in ("salicylic", "aspirin", "paracetamol"):
        source = f"{TEMPLATES}/{molecule}_minima.xyz"
        ktn, coords = network_of(molecule, source)
        pairs, limit = dihedral_enumeration(ktn, coords, neighbours=2,
                                            report_limit=True)
        graph = nx.Graph()
        graph.add_nodes_from(range(ktn.n_minima))
        graph.add_edges_from(pairs)
        assert nx.number_connected_components(graph) == 1, (
            f"{molecule} left {nx.number_connected_components(graph)} "
            f"components at a limit of {limit}")


def test_pairs_are_unique_and_ordered():
    ktn, coords = network_of("salicylic",
                             f"{TEMPLATES}/salicylic_minima.xyz")
    pairs = dihedral_enumeration(ktn, coords, neighbours=2)
    assert all(i < j for i, j in pairs)
    assert len(pairs) == len({tuple(p) for p in pairs})
