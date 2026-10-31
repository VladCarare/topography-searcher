import pytest
import numpy as np
from numpy.testing import assert_array_equal
import ase.io
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.similarity.molecular_similarity import MolecularSimilarity
from topsearch.potentials.test_functions import Schwefel, Camelback
from topsearch.potentials.dft import DensityFunctionalTheory
from topsearch.potentials.force_fields import MMFF94
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.data.coordinates import StandardCoordinates, MolecularCoordinates
from topsearch.global_optimisation.basin_hopping import BasinHopping, \
        MIN_ATOM_SEPARATION
from topsearch.minimisation import lbfgs
from topsearch.global_optimisation.perturbations import StandardPerturbation, \
        MolecularPerturbation
from topsearch.analysis.minima_properties import get_minima_energies

def test_batch_finds_all_minima():
    step_taking = StandardPerturbation(max_displacement=0.7,
                                       proportional_distance=True)
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    camelback = Camelback()
    basin_hopping = BasinHopping(ktn=ktn, potential=camelback,
                                 similarity=similarity,
                                 step_taking=step_taking)
    starting_points = np.array([[0.08984199, -0.7126564 ],
                                [-0.08984203, 0.71265641],
                                [1.60710479, 0.56865138],
                                [-1.70360672, 0.79608357],
                                [ 1.70360672, -0.79608357],
                                [-1.60710476, -0.56865144]]) - 0.05

    basin_hopping.run_batch(initial_positions=starting_points, coords=coords, n_steps=5,
                      temperature=1.0, conv_crit=1e-6)
    assert ktn.n_minima == 6
    energies = get_minima_energies(ktn)
    assert sorted(energies)[0] == pytest.approx(-1.0316284534898734)
    assert sorted(energies)[1] == pytest.approx(-1.0316284534898585)

def test_batch_attempted_positions_to_ktn():
    step_taking = StandardPerturbation(max_displacement=0.7,
                                       proportional_distance=True)
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    camelback = Camelback()
    basin_hopping = BasinHopping(ktn=ktn, potential=camelback,
                                 similarity=similarity,
                                 step_taking=step_taking)
    starting_points = np.array([[0.08984199, -0.7126564 ],
                                [-0.08984203, 0.71265641]]) - 0.05
    basin_hopping.run_batch(initial_positions=starting_points, coords=coords, n_steps=5,
                      temperature=1.0, conv_crit=1e-6)
    attempted = np.array(ktn.get_attempted_positions())
    
    assert_array_equal(attempted[attempted[:,1].argsort()], starting_points)

def test_run_camelback():
    step_taking = StandardPerturbation(max_displacement=0.7,
                                       proportional_distance=True)
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    ktn = KineticTransitionNetwork()
    camelback = Camelback()
    basin_hopping = BasinHopping(ktn=ktn, potential=camelback,
                                 similarity=similarity,
                                 step_taking=step_taking)
    basin_hopping.run(coords=coords, n_steps=1500,
                      temperature=1.0, conv_crit=1e-6)
    energies = get_minima_energies(ktn)
    # Check that we locate all six minima of the function within bounds
    assert sorted(energies)[0] == pytest.approx(-1.0316284534898734)
    assert sorted(energies)[1] == pytest.approx(-1.0316284534898585)

def test_run_schwefel():
    step_taking = StandardPerturbation(max_displacement=0.3,
                                       proportional_distance=True)
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=3, bounds=[(-500.0, 500.0),
                                                 (-500.0, 500.0),
                                                 (-500.0, 500.0)])
    ktn = KineticTransitionNetwork()
    schwefel = Schwefel()
    basin_hopping = BasinHopping(ktn=ktn, potential=schwefel,
                                 similarity=similarity,
                                 step_taking=step_taking)
    basin_hopping.run(coords=coords, n_steps=2500,
                      temperature=100.0, conv_crit=1e-5)
    energies = get_minima_energies(ktn)
    min_energy = min(energies)
    assert min_energy < 1e-3

def test_metropolis():
    step_taking = StandardPerturbation(max_displacement=0.3,
                                       proportional_distance=True)
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0),
                                                 (-2.0, 2.0)])
    coords.position = np.array([0.1, -0.7])
    ktn = KineticTransitionNetwork()
    camelback = Camelback()
    basin_hopping = BasinHopping(ktn=ktn, potential=camelback,
                                 similarity=similarity,
                                 step_taking=step_taking)
    accept = basin_hopping.metropolis(10.0, 9.0, 1.0)
    assert accept == True
    accept = basin_hopping.metropolis(10.0, 8.0, 10.0)
    assert accept == True
    accept = basin_hopping.metropolis(10.0, 11.0, np.inf)
    assert accept == True
    accept = basin_hopping.metropolis(10.0, 11.0, 1e-5)
    assert accept == False

def test_prepare_initial_coordinates():
    step_taking = StandardPerturbation(max_displacement=0.3,
                                       proportional_distance=True)
    similarity = StandardSimilarity(0.1, 0.1)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0),
                                                 (-2.0, 2.0)])
    coords.position = np.array([0.1, -0.7])
    ktn = KineticTransitionNetwork()
    camelback = Camelback()
    basin_hopping = BasinHopping(ktn=ktn, potential=camelback,
                                 similarity=similarity,
                                 step_taking=step_taking)
    energy = basin_hopping.prepare_initial_coordinates(coords, 1e-6)
    assert energy == pytest.approx(-1.031628422)
    assert np.all(coords.position == pytest.approx(np.array([0.0898420385,
                                                             -0.712656397])))


class _CountingPotential:
    """ Harmonic restraint to a reference geometry, counting gradient calls.
        Restraining rather than driving to the origin keeps the minimum at a
        sensible molecular geometry instead of collapsing every atom onto one
        point. """

    def __init__(self, reference):
        self.reference = np.asarray(reference, dtype=float)
        self.calls = 0

    def function_gradient(self, position):
        self.calls += 1
        displacement = np.asarray(position) - self.reference
        return float(np.dot(displacement, displacement)), 2.0*displacement


class _CollapseTwoAtoms:
    """ Perturbation that drops atom 1 on top of atom 0. """

    def __init__(self):
        self.perturbations = 0

    def perturb(self, coords):
        self.perturbations += 1
        position = coords.position.copy()
        position[3:6] = position[0:3] + np.array([0.3, 0.0, 0.0])
        coords.position = position


def _water_coordinates():
    return MolecularCoordinates(
        ['O', 'H', 'H'],
        np.array([0.0, 0.0, 0.0,
                  0.9572, 0.0, 0.0,
                  -0.2399, 0.9267, 0.0]))


def test_closest_approach_is_the_smallest_interatomic_distance():
    coords = _water_coordinates()
    basin_hopping = BasinHopping(ktn=KineticTransitionNetwork(),
                                 potential=_CountingPotential(coords.position),
                                 similarity=MolecularSimilarity(0.1, 0.1),
                                 step_taking=_CollapseTwoAtoms())
    # The two O-H bonds are 0.9572 A; the H...H separation is longer
    assert basin_hopping.closest_approach(coords) == pytest.approx(0.9572)


def test_closest_approach_ignores_non_atomic_coordinates():
    basin_hopping = BasinHopping(ktn=KineticTransitionNetwork(),
                                 potential=Camelback(),
                                 similarity=StandardSimilarity(0.1, 0.1),
                                 step_taking=StandardPerturbation(
                                     max_displacement=0.7,
                                     proportional_distance=True))
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    assert basin_hopping.closest_approach(coords) == np.inf


def _run_collapsing(n_steps):
    """ Basin hopping whose every perturbation clashes two atoms, returning
        the gradient call count and the number of perturbations made. """
    coords = _water_coordinates()
    potential = _CountingPotential(coords.position)
    step_taking = _CollapseTwoAtoms()
    basin_hopping = BasinHopping(ktn=KineticTransitionNetwork(),
                                 potential=potential,
                                 similarity=MolecularSimilarity(0.1, 0.1),
                                 step_taking=step_taking)
    basin_hopping.run(coords=coords, n_steps=n_steps, conv_crit=1e-6,
                      temperature=1.0)
    return potential.calls, step_taking.perturbations, coords, basin_hopping


def test_clashed_perturbation_is_rejected_without_minimising():
    # Every perturbation collapses two atoms to 0.3 A, inside the threshold
    assert MIN_ATOM_SEPARATION > 0.3
    without_steps, no_perturbations, _, _ = _run_collapsing(0)
    with_steps, perturbations, _, _ = _run_collapsing(4)

    assert no_perturbations == 0
    assert perturbations == 4
    assert without_steps > 0, "the initial minimisation should use the potential"
    # Four clashed steps cost nothing beyond that initial minimisation
    assert with_steps == without_steps


def test_clashed_perturbation_is_reverted_to_the_markov_state():
    _, no_perturbations, untouched, _ = _run_collapsing(0)
    _, perturbations, coords, basin_hopping = _run_collapsing(2)
    assert no_perturbations == 0
    assert perturbations == 2
    # The rejected clash must not be left behind in the coordinates, and the
    # revert must restore the Markov state exactly rather than a relaxation of
    # the clash that happens to land nearby
    assert basin_hopping.closest_approach(coords) > MIN_ATOM_SEPARATION
    assert_array_equal(coords.position, untouched.position)


def test_minimisation_step_cap_is_forwarded(monkeypatch):
    recorded = {}
    original = lbfgs.minimise

    def recording_minimise(*args, **kwargs):
        recorded['n_steps'] = kwargs.get('n_steps')
        return original(*args, **kwargs)

    monkeypatch.setattr(lbfgs, 'minimise', recording_minimise)
    basin_hopping = BasinHopping(ktn=KineticTransitionNetwork(),
                                 potential=Camelback(),
                                 similarity=StandardSimilarity(0.1, 0.1),
                                 step_taking=StandardPerturbation(
                                     max_displacement=0.7,
                                     proportional_distance=True),
                                 max_minimisation_steps=37)
    coords = StandardCoordinates(ndim=2, bounds=[(-3.0, 3.0), (-2.0, 2.0)])
    basin_hopping.run(coords=coords, n_steps=1, conv_crit=1e-6, temperature=1.0)
    assert recorded['n_steps'] == 37


def test_minimisation_step_cap_defaults_to_two_hundred():
    basin_hopping = BasinHopping(ktn=KineticTransitionNetwork(),
                                 potential=Camelback(),
                                 similarity=StandardSimilarity(0.1, 0.1),
                                 step_taking=StandardPerturbation(
                                     max_displacement=0.7,
                                     proportional_distance=True))
    assert basin_hopping.max_minimisation_steps == 200


def _wrecked(position):
    """ The sort of geometry a diverged minimisation returns: two atoms on
        top of each other. """
    wreck = position.copy()
    wreck[3:6] = wreck[0:3]
    return wreck


def test_failed_initial_minimisation_is_not_adopted(monkeypatch):
    coords = _water_coordinates()
    start = coords.position.copy()
    potential = _CountingPotential(start)

    def diverging_minimise(*args, **kwargs):
        return _wrecked(start), -1.0e9, {'warnflag': 2, 'task': 'ABNORMAL', 'nit': 234}

    monkeypatch.setattr(lbfgs, 'minimise', diverging_minimise)
    ktn = KineticTransitionNetwork()
    basin_hopping = BasinHopping(ktn=ktn, potential=potential,
                                 similarity=MolecularSimilarity(0.1, 0.1),
                                 step_taking=_CollapseTwoAtoms())
    energy = basin_hopping.prepare_initial_coordinates(coords, conv_crit=1e-6)

    # A diverged minimisation must not become the Markov state that every
    # later step is perturbed from
    assert_array_equal(coords.position, start)
    assert basin_hopping.closest_approach(coords) > MIN_ATOM_SEPARATION
    # Nor may it enter the network as a minimum
    assert ktn.n_minima == 0
    # The energy returned has to belong to the coordinates that were kept
    assert energy == pytest.approx(potential.function_gradient(start)[0])


def test_converged_initial_minimisation_is_adopted(monkeypatch):
    coords = _water_coordinates()
    start = coords.position.copy()
    moved = start + 0.01

    def converging_minimise(*args, **kwargs):
        return moved, -7.0, {'warnflag': 0, 'task': 'CONVERGENCE', 'nit': 12}

    monkeypatch.setattr(lbfgs, 'minimise', converging_minimise)
    ktn = KineticTransitionNetwork()
    basin_hopping = BasinHopping(ktn=ktn, potential=_CountingPotential(start),
                                 similarity=MolecularSimilarity(0.1, 0.1),
                                 step_taking=_CollapseTwoAtoms())
    energy = basin_hopping.prepare_initial_coordinates(coords, conv_crit=1e-6)

    assert_array_equal(coords.position, moved)
    assert ktn.n_minima == 1
    assert energy == pytest.approx(-7.0)
