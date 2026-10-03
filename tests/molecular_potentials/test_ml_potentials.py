import os
import pytest
import numpy as np
import ase.io
from ase.units import Hartree
from topsearch.potentials.ml_potentials import MachineLearningPotential

# torchani reports in Hartree. Every other calculator reaches topsearch
# through ase and reports in eV, so the torchani branch converts and these
# references are written as the Hartree value times the conversion.
ANI_ETHANOL_HARTREE = -154.98837187806524

current_dir = os.path.dirname(os.path.dirname((os.path.realpath(__file__))))

# Every calculator here needs torch, which is an optional dependency
# (the ani_potential group).
torch = pytest.importorskip("torch", reason="ani_potential extra not installed")

def test_ani_function():
    atoms = ase.io.read(f'{current_dir}/test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    mlp = MachineLearningPotential(atom_labels=species,
                                   calculator_type='torchani')
    energy = mlp.function(position)
    assert energy == pytest.approx(ANI_ETHANOL_HARTREE * Hartree)

def test_ani_gradient():
    atoms = ase.io.read(f'{current_dir}/test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    mlp = MachineLearningPotential(atom_labels=species,
                                   calculator_type='torchani')
    grad = mlp.gradient(position)
    assert np.all(np.abs(grad) < 5e-2 * Hartree)

def test_ani_function_gradient():
    atoms = ase.io.read(f'{current_dir}/test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    mlp = MachineLearningPotential(atom_labels=species,
                                   calculator_type='torchani')
    energy, grad = mlp.function_gradient(position)
    assert energy == pytest.approx(ANI_ETHANOL_HARTREE * Hartree)
    assert np.all(np.abs(grad) < 5e-2 * Hartree)


def test_force_field_is_stored():
    """ The ff kwarg feeds BasinHopping's clash removal for molecules """
    atoms = ase.io.read(f'{current_dir}/test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    sentinel = object()
    mlp = MachineLearningPotential(atom_labels=species,
                                   calculator_type='torchani',
                                   ff=sentinel)
    assert mlp.force_field is sentinel


def test_force_field_defaults_to_none():
    atoms = ase.io.read(f'{current_dir}/test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    mlp = MachineLearningPotential(atom_labels=species,
                                   calculator_type='torchani')
    assert mlp.force_field is None


def test_torchani_energy_is_converted_from_hartree():
    """ The torchani branch must return eV, like every other calculator.
        Without the conversion it returns the raw Hartree value, which is
        silently wrong by a factor of 27.2 against the rest of the package. """
    atoms = ase.io.read(f'{current_dir}/test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    mlp = MachineLearningPotential(atom_labels=species,
                                   calculator_type='torchani')
    energy = mlp.function(position)
    assert energy == pytest.approx(ANI_ETHANOL_HARTREE * Hartree)
    assert energy != pytest.approx(ANI_ETHANOL_HARTREE)


def test_unknown_calculator_is_rejected():
    atoms = ase.io.read(f'{current_dir}/test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    with pytest.raises(Exception):
        MachineLearningPotential(atom_labels=species,
                                 calculator_type='not-a-real-model')


def test_energy_is_a_scalar():
    """ Some ase calculators report the energy as a one element array.
        KineticTransitionNetwork.dump_network builds a ragged array from a
        mix of scalars and arrays and fails with "setting an array element
        with a sequence", so the potential must hand back a number. """
    atoms = ase.io.read(f'{current_dir}/test_data/ethanol.xyz')
    species = atoms.get_chemical_symbols()
    position = atoms.get_positions().flatten()
    mlp = MachineLearningPotential(atom_labels=species,
                                   calculator_type='torchani')
    assert isinstance(mlp.function(position), float)
    energy, _ = mlp.function_gradient(position)
    assert isinstance(energy, float)


def test_as_energy_accepts_array_and_scalar():
    """ Covers the AIMNet2 shape without needing AIMNet2 installed """
    from topsearch.potentials.ml_potentials import _as_energy
    assert _as_energy(np.array([-13506.94237821])) == pytest.approx(-13506.94237821)
    assert _as_energy(-13506.94237821) == pytest.approx(-13506.94237821)
    assert isinstance(_as_energy(np.array([1.0])), float)
