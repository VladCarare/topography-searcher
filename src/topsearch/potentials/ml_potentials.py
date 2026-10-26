""" Module that contains the class for evaluating the energy of molecules
    via machine-learned potentials through the ase package """

import os

import numpy as np
from ase.calculators.calculator import CalculatorError
from numpy.typing import NDArray
from ase import Atoms
import warnings
from ase.units import Hartree
from .potential import Potential



def _as_energy(value) -> float:
    """ Return a scalar energy.

        Some ase calculators, AIMNet2 among them, report the energy as a
        one element array. The rest of topsearch stores it in the network
        and compares it as a number, and np.append builds a ragged array
        from a mix of the two, so normalise here where the calculator is
        known rather than guarding at every use.
    """
    return float(np.asarray(value).reshape(-1)[0])


# Energy reported when a calculator cannot evaluate a structure at all.
# A semi-empirical or first-principles method solves an electronic
# structure problem and can legitimately fail to converge on a distorted
# geometry; the structure is then unreachable rather than the run being
# broken, and a large finite energy says exactly that to whatever is
# searching. Fitted models never refuse, so this only arises for the
# physics-based calculators.
UNREACHABLE_ENERGY = 1e6
# the force field's harmonic terms reach 1e6 eV/A on a torn
# structure, so the fallback direction is kept and the size clamped
FALLBACK_MAX_GRADIENT = float(os.environ.get(
    "MLP_FALLBACK_MAX_GRADIENT", "10.0"))


class MachineLearningPotential(Potential):
    """
    Description
    ---------------
    Evaluate the energy of a molecular system using machine learning potentials
    called through the ase package. Options are ANI2x and MACE.

    Attributes
    ---------------
    atom_labels : list
        List of atomic species for each atom in the molecule.
    calculator_type : str
        Type of calculator to use. Options are 'mace' or 'torchani'
    """
    def __init__(self, 
                 atom_labels: list,
                 calculator_type: str = 'torchani',
                 model: str = 'default',
                 device: str = 'cpu',
                 ff = None):
    
        self.atomistic = True
        self.calculator_type = calculator_type
        self.atom_labels = atom_labels
        self.device = device
        self.force_field = ff
        # Make a placeholder atomic configuration for initialising calculator
        n_atoms = len(self.atom_labels)
        init_position = np.ones((n_atoms, 3), dtype=float) * \
            np.arange(0, n_atoms, dtype=float).reshape(-1, 1)
        # Create the atoms object for the molecule
        self.atoms = Atoms(''.join(self.atom_labels),
                           positions=init_position)
        self.calculator_type = calculator_type
        # Set up the calculator based on the specified type
        # how many structures the calculator could not evaluate
        self.failed_evaluations = 0
        if self.calculator_type == 'torchani':
            import torchani, torch
            torch.set_default_dtype(torch.float64)
            self.model = torchani.models.ANI2x(periodic_table_index=True)
            self.atoms.calc = \
                torchani.ase.Calculator(list('HCNOFSCl'), self.model)
            if self.device == 'mps':
                self.model.to(torch.float32)
            self.model.to(self.device)
        elif self.calculator_type == 'mace':
            if model == 'default':
                model = ['MACE_model_swa.model']
            from mace.calculators import MACECalculator
            import torch
            torch.set_default_dtype(torch.float64)
            self.atoms.calc = \
                MACECalculator(model_paths=model,
                               device=device)
        elif self.calculator_type == 'mace-mp-0b3':
            from mace.calculators import mace_mp 
            model_path = 'mace-mp-0b3-medium.model'
            print(f'Using model at: {model_path}')
            import torch
            torch.set_default_dtype(torch.float64)
            self.atoms.calc = \
                mace_mp(model=model_path,dispersion=False, default_dtype="float64",
                               device=device)
        elif self.calculator_type == 'nequip':
            import torch
            torch.set_default_dtype(torch.float64)
            print("setting default to float64")
            from nequip.ase import NequIPCalculator
            self.atoms.calc = \
                NequIPCalculator.from_deployed_model(model,
                               device=device,
                               set_global_options=False)
        elif self.calculator_type == 'so3lr':
            import jax
            jax.config.update("jax_enable_x64", True)
            from so3lr import So3lrCalculator
            self.atoms.calc = \
                So3lrCalculator(
                calculate_stress=False,
                dtype=np.float64
            )
        elif self.calculator_type == 'aimnet2':
            import torch
            torch.set_default_dtype(torch.float64)
            from aimnet2calc import AIMNet2ASE
            if model == 'default':
                model = 'aimnet2'
            if device != 'cpu' and device != 'cuda':
                warnings.warn("Aimnet2 ignores 'device' argument")
            self.atoms.calc = \
                AIMNet2ASE(model)
        elif self.calculator_type == 'uma':
            # UMA reads charge and spin from atoms.info and defaults them
            # silently, so an uncharged closed-shell molecule has to say so
            # explicitly rather than rely on the default being right.
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
            if model == 'default':
                model = 'uma-s-1p1'
            predictor = pretrained_mlip.get_predict_unit(
                model, device='cuda' if device == 'cuda' else 'cpu')
            self.atoms.info.update({'charge': 0, 'spin': 1})
            self.atoms.calc = FAIRChemCalculator(predictor, task_name='omol')
        elif self.calculator_type == 'gfn2-xtb':
            # Semi-empirical rather than fitted, so it serves as a control:
            # a landscape it explores should contain no structures that
            # chemistry forbids, whatever the search does.
            from tblite.ase import TBLite
            self.atoms.calc = TBLite(method='GFN2-xTB', verbosity=0)
        else:
            raise ValueError(
                f"Unknown calculator_type {calculator_type!r}. Expected one "
                "of: torchani, mace, mace-mp-0b3, nequip, so3lr, aimnet2, "
                "gfn2-xtb, uma")

    def _rebuild_calculator(self) -> None:
        """Replace the calculator after it declined to evaluate a structure.

        tblite does not fully recover from a failed calculation: a later
        call aborts the process from inside its Fortran layer, which no
        Python handler can catch. Rebuilding after each refusal keeps a
        search alive, and costs nothing because refusals are rare on
        structures that are not badly distorted.
        """
        if self.calculator_type == 'gfn2-xtb':
            from tblite.ase import TBLite
            self.atoms.calc = TBLite(method='GFN2-xTB', verbosity=0)

    def _fallback(self, position: NDArray) -> tuple:
        """Energy and gradient for a structure the model declined.

        The force field stands in, shifted so the point stays
        unattractive. Without a force field the gradient is zero, which
        a minimiser reads as converged, so the structure can be stored
        as a spurious minimum.
        """
        if self.force_field is None:
            return UNREACHABLE_ENERGY, np.zeros_like(position)
        energy, gradient = self.force_field.function_gradient(position)
        gradient = np.asarray(gradient, dtype=float)
        largest = np.abs(gradient).max()
        if largest > FALLBACK_MAX_GRADIENT:
            gradient = gradient * (FALLBACK_MAX_GRADIENT / largest)
        return UNREACHABLE_ENERGY + float(energy), gradient

    def function(self, position: NDArray) -> float:
        """ Compute the electronic potential energy.

            A calculator that cannot converge on this geometry reports the
            structure as unreachable rather than raising, so that one bad
            structure does not end a search. Only non-convergence is caught:
            any other error is a genuine fault and still propagates.
        """
        self.atoms.set_positions(position.reshape(-1, 3))
        if self.calculator_type in ['mace', 'aimnet2', 'nequip',
                                   'mace-mp-0b3', 'so3lr',
                                   'gfn2-xtb', 'uma']:
            try:
                energy = self.atoms.get_potential_energy()
            except CalculatorError:
                self.failed_evaluations += 1
                self._rebuild_calculator()
                return self._fallback(position)[0]
        elif self.calculator_type == 'torchani':
            import torch
            species = torch.tensor(self.atoms.get_atomic_numbers(),
                                   dtype=torch.int64).unsqueeze(0).to(self.device)
            coordinates = torch.tensor(position.reshape(-1, 3),
                                       dtype=torch.float32,
                                       requires_grad=True).unsqueeze(0).to(self.device)
            energy = self.model((species, coordinates)).energies.item() * Hartree
        return _as_energy(energy)

    def function_gradient(self, position: NDArray) -> tuple:
        """ Compute the electronic potential energy and its forces """
        self.atoms.set_positions(position.reshape(-1, 3))

        if self.calculator_type in ['mace', 'mace-mp-0b3', 'so3lr',
                                    'gfn2-xtb', 'uma']:
            try:
                forces = self.atoms.get_forces().flatten()
                energy = self.atoms.get_potential_energy()
            except CalculatorError:
                self.failed_evaluations += 1
                self._rebuild_calculator()
                return self._fallback(position)
            gradient = -1.0 * np.array(forces.tolist())
        elif self.calculator_type in ['aimnet2', 'nequip']:
            self.atoms.calc.calculate(self.atoms, properties=['energy', 'forces'])
            forces = self.atoms.get_forces().flatten()
            energy = self.atoms.get_potential_energy()
            gradient = -1.0 * np.array(forces.tolist())
        elif self.calculator_type == 'torchani':
            import torch
            species = torch.tensor(self.atoms.get_atomic_numbers(),
                                   dtype=torch.int64).unsqueeze(0)
            coordinates = torch.tensor(position.reshape(-1, 3),
                                       dtype=torch.float32,
                                       requires_grad=True).unsqueeze(0)
            energy_torch = self.model((species, coordinates)).energies * Hartree
            gradient_torch = torch.autograd.grad(energy_torch.sum(),
                                                 coordinates)[0]
            energy = energy_torch.item()
            gradient = np.array(gradient_torch.flatten().tolist())
        return _as_energy(energy), gradient

    def gradient(self, position: NDArray) -> NDArray:
        """ Compute the analytical gradient from the ASE calculator.

            A calculator that cannot converge on this geometry reports a
            zero gradient alongside the unreachable energy its energy
            method returns, so that one bad structure does not end a
            search. Numerical second derivatives go through here, so
            without this a Hessian anywhere near such a geometry fails.
        """
        self.atoms.set_positions(position.reshape(-1, 3))
        if self.calculator_type in ['mace', 'aimnet2', 'nequip',
                                   'mace-mp-0b3', 'so3lr',
                                   'gfn2-xtb', 'uma']:
            try:
                forces = self.atoms.get_forces().flatten()
            except CalculatorError:
                self.failed_evaluations += 1
                self._rebuild_calculator()
                return self._fallback(position)[1]
            gradient = -1.0 * np.array(forces.tolist())
        elif self.calculator_type == 'torchani':
            import torch
            species = torch.tensor(self.atoms.get_atomic_numbers(),
                                   dtype=torch.int64).unsqueeze(0)
            coordinates = torch.tensor(position.reshape(-1, 3),
                                       dtype=torch.float32,
                                       requires_grad=True).unsqueeze(0)
            energy_torch = self.model((species, coordinates)).energies * Hartree
            gradient_torch = torch.autograd.grad(energy_torch.sum(),
                                                 coordinates)[0]
            gradient = np.array(gradient_torch.flatten().tolist())
        return gradient
