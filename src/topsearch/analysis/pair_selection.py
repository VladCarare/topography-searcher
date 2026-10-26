""" Routines that act on a KineticTransitionNetwork object to select pairs 
    of minima for sampling by different criteria """

import logging
import numpy as np
from numpy.typing import NDArray
import networkx as nx
from topsearch.data.coordinates import StandardCoordinates

from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.similarity.similarity import StandardSimilarity
from .graph_properties import unconnected_component
from .minima_properties import get_distance_matrix, get_distance_from_minimum
logger = logging.getLogger()

def connect_unconnected(ktn: KineticTransitionNetwork, similarity: StandardSimilarity,
                        coords: StandardCoordinates, neighbours: int) -> list:
    """
    Find all minima not connected to global minimum set and their nearest
    neighbours. Return the list of minima pairs
    """
    # Check for emptiness
    if ktn.n_minima == 0:
        return []
    # Get the set of minima not connected to the global minimum
    unconnected_set = unconnected_component(ktn)
    total_pairs = []
    if len(unconnected_set) > 0:
        for i in unconnected_set:
            pairs = connect_to_set(ktn, similarity, coords, i, neighbours)
            for j in pairs:
                total_pairs.append(j)
    # Can generate a lot of repeats so remove any repeated pairs
    total_pairs = unique_pairs(total_pairs)
    return total_pairs


def connect_to_set(ktn: KineticTransitionNetwork, similarity: StandardSimilarity, coords: StandardCoordinates,
                   node1: int, cycles: int) -> list:
    """
    Finds all minima connected to node1 and finds the pairs closest
    in distance where one is connected and one is not. Returns the
    set of minima pairs in a list for use in connect_unconnected
    """

    # Set of nodes connected to node1
    s_set = nx.node_connected_component(ktn.G, node1)
    # Find list of nodes not connected to node 1
    f_set = set(range(ktn.n_minima)) - set(s_set)
    if f_set == set():
        logger.info("No unconnected minima\n")
        return []
    # Get ordered distances to this node
    dist_vector = get_distance_from_minimum(ktn, similarity, coords, node1)
    nearest = np.argsort(dist_vector).tolist()[1:]
    # Remove any minima in same set
    pairs = [i for i in nearest if i in f_set]
    total_pairs = []
    for i in pairs[:cycles]:
        total_pairs.append([node1, i])
    return unique_pairs(total_pairs)


def closest_enumeration(ktn: KineticTransitionNetwork, similarity: StandardSimilarity,
                        coords: StandardCoordinates, neighbours: int) -> list:
    """
    Selector that attempts to connect all minima in the fewest number of
    attempts by connecting each minimum to its N nearest neighbours.
    Returns a list of pairs
    """
    pairs = []
    dist_matrix = get_distance_matrix(ktn, similarity, coords)
    for i in range(ktn.n_minima):
        nearest = np.argsort(dist_matrix[i, :]).tolist()[1:neighbours+1]
        for j in nearest:
            pairs.append([i, j])
    return unique_pairs(pairs)


def read_pairs(text_path: str = ''):
    """ Read the set of pairs from the file pairs.txt """
    pairs = np.genfromtxt(f'{text_path}pairs.txt', dtype=int)
    return unique_pairs(pairs.tolist())


def unique_pairs(initial_pairs: list) -> list:
    """ Remove any repeated pairs from a given list """
    # Sort the pairs as [0, 1] and [1, 0] are equivalent
    final_pairs = [tuple(sorted(i)) for i in initial_pairs if i != [0, 0]]
    return [list(i) for i in set(final_pairs)]

def rotatable_dihedrals(coords: StandardCoordinates) -> list:
    """ The proper dihedrals of a molecule, one per bond that can turn.

        A bond can turn if both of its atoms carry another neighbour, so that
        there is something on each side to measure the angle between. Rings
        are included, although rotating through one is not a motion the
        molecule can make; they cost a little accuracy in the separation
        count rather than breaking it.
    """
    import networkx as nx
    from ase import Atoms
    from ase.neighborlist import natural_cutoffs, NeighborList

    labels = getattr(coords, 'atom_labels', None)
    if labels is None:
        return []
    atoms = Atoms(labels, np.asarray(coords.position).reshape(-1, 3))
    neighbours = NeighborList(natural_cutoffs(atoms, mult=1.15), skin=0.0,
                              self_interaction=False, bothways=True)
    neighbours.update(atoms)
    bonds = nx.Graph()
    bonds.add_nodes_from(range(len(atoms)))
    for i in range(len(atoms)):
        for j in neighbours.get_neighbors(i)[0]:
            bonds.add_edge(int(i), int(j))
    dihedrals = []
    for b, c in bonds.edges:
        before = [n for n in bonds.neighbors(b) if n != c]
        after = [n for n in bonds.neighbors(c) if n != b]
        if before and after:
            dihedrals.append((before[0], b, c, after[0]))
    return dihedrals


def dihedral_separation(coords: StandardCoordinates, position1: NDArray,
                        position2: NDArray, dihedrals: list,
                        threshold: float = 30.0) -> int:
    """ How many dihedrals have to turn to get from one minimum to the other.

        Counts rotations rather than measuring them: a bond that has turned
        by 90 degrees and one that has turned by 180 both count once, because
        the interpolation has to carry out one rotation either way. What
        matters for the interpolation is how many happen at once.
    """
    from ase import Atoms

    labels = coords.atom_labels
    first = Atoms(labels, np.asarray(position1).reshape(-1, 3))
    second = Atoms(labels, np.asarray(position2).reshape(-1, 3))
    turned = 0
    for dihedral in dihedrals:
        try:
            change = (second.get_dihedral(*dihedral)
                      - first.get_dihedral(*dihedral) + 180.0) % 360.0 - 180.0
        except (ValueError, ZeroDivisionError):
            continue
        if abs(change) > threshold:
            turned += 1
    return turned


def dihedral_enumeration(ktn: KineticTransitionNetwork,
                         coords: StandardCoordinates, neighbours: int = 2,
                         report_limit: bool = False):
    """ Selector that connects each minimum to its nearest in dihedral space.

        closest_enumeration ranks neighbours by distance, which only loosely
        tracks how many rotations lie between two minima. Measured on
        aspirin, 45% of the pairs it picks are separated by three dihedrals,
        and the dihedral interpolation has to perform all three at once; the
        chance of such an attempt producing an off-reference structure is 16%
        against 0% for one rotation and 2% for two.

        Ranking by separation instead keeps every interpolation short. No
        minimum is stranded, because a pair that is no longer attempted
        directly is still reachable through intermediates: the threshold is
        raised until the graph of selected pairs is connected, which needed
        one rotation for salicylic acid, two for aspirin and malonaldehyde,
        and three for paracetamol.

        The hypothesis did not survive the comparison. Against
        closest_enumeration over 100 landscapes, 5 molecules x 2 models x 5
        seeds, minima recovery was unchanged, transition-state recovery was
        unchanged for GFN2-xTB and 8% worse for AIMNet2, and the off-bonding
        structures this was written to remove stayed at 5 on aspirin. What
        removed them was choosing the rotation direction by clearance, from 5
        to 0. Those pairs were hard because one rotation swept a group
        through the molecule, not because three happened at once.
    """
    import networkx as nx

    dihedrals = rotatable_dihedrals(coords)
    if not dihedrals:
        return ([], 0) if report_limit else []
    separations = {}
    for i in range(ktn.n_minima):
        for j in range(i + 1, ktn.n_minima):
            separations[(i, j)] = dihedral_separation(
                coords, ktn.get_minimum_coords(i),
                ktn.get_minimum_coords(j), dihedrals)

    # Raise the threshold only as far as connectivity demands, so that the
    # long moves are attempted only where nothing shorter reaches.
    limit, reachable = 1, len(dihedrals)
    while limit <= reachable:
        graph = nx.Graph()
        graph.add_nodes_from(range(ktn.n_minima))
        graph.add_edges_from(pair for pair, n in separations.items()
                             if 0 < n <= limit)
        if nx.number_connected_components(graph) == 1:
            break
        limit += 1

    pairs = []
    for i in range(ktn.n_minima):
        candidates = sorted(
            ((separations[tuple(sorted((i, j)))], j)
             for j in range(ktn.n_minima) if j != i),
            key=lambda item: item[0])
        for separation, j in candidates[:neighbours]:
            if 0 < separation <= limit:
                pairs.append([min(i, j), max(i, j)])
    # Whatever connectivity the nearest-neighbour pass missed, add directly.
    graph = nx.Graph()
    graph.add_nodes_from(range(ktn.n_minima))
    graph.add_edges_from(tuple(p) for p in pairs)
    for pair, n in sorted(separations.items(), key=lambda item: item[1]):
        if nx.number_connected_components(graph) == 1:
            break
        if 0 < n <= limit and not nx.has_path(graph, *pair):
            pairs.append([pair[0], pair[1]])
            graph.add_edge(*pair)

    pairs = unique_pairs(pairs)
    return (pairs, limit) if report_limit else pairs
