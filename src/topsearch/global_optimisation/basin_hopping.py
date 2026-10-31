""" The basin_hopping module contains the BasinHopping class which performs
    global optimisation of a given function """

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import logging
from timeit import default_timer as timer
import numpy as np
from topsearch.data.kinetic_transition_network import KineticTransitionNetwork
from topsearch.global_optimisation.perturbations import AtomicPerturbation, MolecularPerturbation, StandardPerturbation
from topsearch.minimisation import lbfgs, psi4_internal
from topsearch.data.coordinates import MolecularCoordinates, AtomicCoordinates, StandardCoordinates
from topsearch.potentials.dft import DensityFunctionalTheory
from topsearch.potentials.potential import Potential
from topsearch.similarity.similarity import StandardSimilarity
from topsearch.utils.parallel import run_parallel
from tqdm.auto import tqdm, trange

#: Reject a perturbed geometry outright if any pair of atoms is closer than
#: this, in Angstrom. Such geometries are unphysical and the minimiser aborts
#: on them immediately, so there is nothing to be gained by relaxing them.
MIN_ATOM_SEPARATION = 0.6

class BasinHopping:

    """
    Description
    ------------

    Class to perform basin-hopping global optimisation algorithm
    as described in https://doi.org/10.1021/jp970984n
    One iteration is described by:
    1) Perturb current position
    2) Minimisation to get corresponding local minimum
    3) Accept/reject based on Metropolis-like criterion

    Attributes
    ----------

    ktn : class instance
        An instance of the KineticTransitionNetwork class that is used
        to store the unique minima we locate
    potential : class instance
        Object containing the function that we are aiming to optimise
    similarity: class instance
        Object that determines if two minima are the same or unique
    step_taking : class instance
        Instance that perturbs the current position using a given set
        of moves specified in the class
    """

    def __init__(self,
                 ktn: KineticTransitionNetwork,
                 potential: Potential,
                 similarity: StandardSimilarity,
                 step_taking: StandardPerturbation | AtomicPerturbation | MolecularPerturbation,
                 opt_method: str = 'scipy',
                 ignore_relreduc: bool = False,
                 max_minimisation_steps: int = 200,
                 record: list | None = None) -> None:
        self.ktn = ktn
        self.potential = potential
        self.similarity = similarity
        self.step_taking = step_taking
        self.logger = logging.getLogger("basin-hopping")
        self.opt_method = opt_method
        self.ignore_relreduc = ignore_relreduc
        self.max_minimisation_steps = max_minimisation_steps
        # Optional per-step log. The caller owns it and whatever it does with
        # it, so the library stays free of file handling; every step lands in
        # it, accepted or rejected, with the reason.
        self.record = record

    def run_batch(self, initial_positions: np.ndarray, coords: type, n_steps: int, conv_crit: float,
                temperature: float, num_proc=8) -> None:
        self.logger.debug(f"Running basin hopping for {len(initial_positions)} starting points with {num_proc} processes")

        run_step = partial(self.run_single, coords=coords, n_steps=n_steps, conv_crit=conv_crit, temperature=temperature)

        for position, ktn in run_parallel(run_step, initial_positions, processes=num_proc, return_input=True):
                for m in range(ktn.n_minima):
                    try:
                        coords.position = ktn.get_minimum_coords(m)
                        self.logger.debug(f"New minimum value {ktn.get_minimum_energy(m)}")
                        self.similarity.test_new_minimum(self.ktn, coords, ktn.get_minimum_energy(m))
                    except KeyError:
                        self.logger.error(f"Missing minimum index {m}. n_minima: {ktn.n_minima} Nodes in graph: {len(ktn.G.nodes)}.\n Keys: {ktn.G.nodes.keys()} ")
                        ktn.dump_network(text_string="bad_minima")
                
                self.ktn.add_attempted_position(position)
                self.ktn.dump_network()
           
    def run_single(self, position: np.ndarray, coords: type, n_steps: int, conv_crit: float,
                temperature: float):
            
            coords.position = position
            self.logger.debug(f"Starting position: {position}")
            self.logger.debug(f"Current KTN with n_minima: {self.ktn.n_minima} nodes in graph: {len(self.ktn.G.nodes)}")
            self.ktn.reset_network() # Clear the KTN to aovid returning a massive one
            self.run(coords=coords, n_steps=n_steps, conv_crit=conv_crit, temperature=temperature)
            self.logger.debug(f"Returning KTN with n_minima: {self.ktn.n_minima} nodes in graph: {len(self.ktn.G.nodes)}")
            return self.ktn

    def run(self, coords: StandardCoordinates | MolecularCoordinates | AtomicCoordinates, n_steps: int, conv_crit: float,
            temperature: float) -> None:
        """ Method to perform basin-hopping from a given start point """
        start_time = timer()
        self.write_initial_information(n_steps, temperature, conv_crit)
        energy = self.prepare_initial_coordinates(coords, conv_crit)
        # coords/energy - coordinates after each perturbation
        # markov_coords/energy - current minimum in the Markov chain
        markov_coords = coords.position.copy()
        markov_energy = energy
        # Main loop for repeated perturbation, minimisation, and acceptance
        for i in trange(n_steps, desc="Basin hopping - steps"):
            self.logger.debug(f"Step {i} of {n_steps}")
            #  Perturb coordinates
            self.step_taking.perturb(coords)
            # Test for and remove atom clashes. The two coordinate classes
            # take different arguments: MolecularCoordinates relaxes the clash
            # with an empirical force field supplied by the potential, while
            # AtomicCoordinates does it geometrically and takes none. Dispatch
            # on the coordinates, since MolecularCoordinates subclasses
            # AtomicCoordinates and would otherwise match the wrong branch.
            force_field = getattr(self.potential, 'force_field', None)
            if isinstance(coords, MolecularCoordinates):
                if force_field is not None:
                    coords.remove_atom_clashes(force_field)
                else:
                    self.logger.debug("No force field on the potential, "
                                      "skipping clash removal")
            elif isinstance(coords, AtomicCoordinates):
                coords.remove_atom_clashes()
            # Reject a geometry whose atoms overlap rather than minimising it
            separation = self.closest_approach(coords)
            if separation < MIN_ATOM_SEPARATION:
                self.logger.info(f"Step {i+1}: rejected before minimisation, "
                                 f"closest atoms {separation:.3f} < "
                                 f"{MIN_ATOM_SEPARATION} Angstrom")
                self.note_step(i, 'clashed', coords.position)
                coords.position = markov_coords
                continue
            # Kept only for the record, so not copied when nobody asked for one
            perturbed = None if self.record is None else coords.position.copy()
            # Perform local minimisation
            if self.opt_method == 'scipy':
                min_position, energy, results_dict = \
                    lbfgs.minimise(func_grad=self.potential.function_gradient,
                                initial_position=coords.position,
                                bounds=coords.bounds,
                                conv_crit=conv_crit,
                                n_steps=self.max_minimisation_steps)
            elif self.opt_method == 'ase':
                min_position, energy, results_dict = \
                    lbfgs.minimise_ase(self.potential,
                                       initial_position=coords.position,
                                       numbers=coords.atoms.numbers,
                                       conv_crit=conv_crit,
                                       n_steps=self.max_minimisation_steps)
            elif self.opt_method == 'psi4':
                if not isinstance(self.potential, DensityFunctionalTheory):
                    raise ValueError("Psi4 optimisation method requires DFT potential")
                min_position, energy, results_dict = \
                psi4_internal.minimise(self.potential,
                                initial_position=coords.position,
                                conv_crit=conv_crit)
            else:
                raise ValueError("Invalid optimisation method, options are 'scipy' or 'psi4'")
            # Failed minimisation so don't accept
            if results_dict['warnflag'] != 0 or \
                    (results_dict['task'] == 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH' \
                        and not self.ignore_relreduc):
                
                self.logger.info(f"Step {i+1}: Could not converge with {results_dict} \n")
                self.note_step(i, 'unconverged', perturbed, min_position,
                               energy, results_dict)
                # Revert to previous coordinates
                coords.position = markov_coords
                continue
            coords.position = min_position
            # coords.write_xyz(f"_bh_{i}_")
            # Check that the bonds have not changed if molecular
            if isinstance(coords, (AtomicCoordinates, MolecularCoordinates)):
                # Bonds have been changed so reject step
                if not coords.same_bonds():
                    # Revert coordinates and reject step
                    self.note_step(i, 'bonding_changed', perturbed,
                                   min_position, energy, results_dict)
                    coords.position = markov_coords
                    continue
            self.note_step(i, 'kept', perturbed, min_position, energy,
                           results_dict)
            # Check the Metropolis criterion for acceptance of new state
            if self.metropolis(markov_energy, energy, temperature):
                # Check similarity and add if different to all existing minima
                self.similarity.test_new_minimum(self.ktn, coords, energy)
                # Update the current minimum in the Markov chain
                markov_coords = coords.position
                markov_energy = energy
            else:
                # Check if minimum should be added even if not accepted
                self.similarity.test_new_minimum(self.ktn, coords, energy)
                coords.position = markov_coords
                energy = markov_energy
        end_time = timer()
        self.write_final_information(start_time, end_time)

    def note_step(self, step: int, verdict: str, before: np.ndarray,
                  after: np.ndarray = None, energy: float = None,
                  results_dict: dict = None) -> None:
        """ Append one basin hopping step to the caller's record, if they
            asked for one. `before` is the perturbed geometry handed to the
            minimiser and `after` what it returned, absent for a step rejected
            before any minimisation. """
        if self.record is None:
            return
        entry = {'step': step, 'verdict': verdict,
                 'perturbed': np.array(before, dtype=float).copy(),
                 'relaxed': None if after is None
                            else np.array(after, dtype=float).copy(),
                 'energy': None if energy is None else float(energy),
                 'warnflag': None if results_dict is None
                             else int(results_dict['warnflag']),
                 'n_iterations': None if results_dict is None
                                 else int(results_dict.get('nit', -1))}
        self.record.append(entry)

    def closest_approach(self, coords) -> float:
        """ Smallest distance between any pair of atoms, in Angstrom.
            Returns infinity for coordinate types that are not atomic. """
        if not isinstance(coords, (AtomicCoordinates, MolecularCoordinates)):
            return np.inf
        positions = coords.position.reshape(-1, 3)
        separations = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=-1)
        np.fill_diagonal(separations, np.inf)
        return float(separations.min())

    def prepare_initial_coordinates(self, coords: StandardCoordinates,
                                    conv_crit: float) -> float:
        """ Generate the initial minimum and set its energy and position """
        self.logger.debug("Initial minimisation")
        if self.opt_method == 'scipy':
            min_position, energy, results_dict = \
                lbfgs.minimise(func_grad=self.potential.function_gradient,
                            initial_position=coords.position,
                            bounds=coords.bounds,
                            conv_crit=conv_crit,
                            n_steps=self.max_minimisation_steps)
            self.logger.debug(f"Initial minima: {min_position}, energy: {energy}, results_dict: {results_dict}")
        elif self.opt_method == 'ase':
                min_position, energy, results_dict = \
                    lbfgs.minimise_ase(self.potential,
                                       initial_position=coords.position,
                                       numbers=coords.atoms.numbers,
                                       conv_crit=conv_crit,
                                       n_steps=self.max_minimisation_steps)
        elif self.opt_method == 'psi4':
            if not isinstance(self.potential, DensityFunctionalTheory):
                raise ValueError("Psi4 optimisation method requires DFT potential")
            min_position, energy, results_dict = \
            psi4_internal.minimise(self.potential,
                            initial_position=coords.position,
                            conv_crit=conv_crit)
        else:
            raise ValueError("Invalid optimisation method, options are 'scipy' or 'psi4'")
        # Only adopt the minimised coordinates when the minimisation actually
        # converged. A diverged one can return a collapsed geometry, and since
        # this becomes the Markov state that every later step is perturbed
        # from, adopting it lets a single bad minimisation poison the whole
        # run: every subsequent step starts from the wreck and is rejected.
        if results_dict['warnflag'] == 0:
            coords.position = min_position
            # After minimisation add to the existing stationary point network
            self.similarity.test_new_minimum(self.ktn, coords, energy)
        else:
            self.logger.warning(
                "Initial minimisation failed with warnflag %s; keeping the "
                "starting coordinates as the Markov state",
                results_dict['warnflag'])
            energy = self.potential.function_gradient(coords.position)[0]
             
        return energy

    def metropolis(self, energy1: float, energy2: float,
                   temperature: float) -> bool:
        """
        Compute the acceptance probability using the Metropolis criterion.
        energy2 - proposed minimum, energy1 - current state in Markov chain
        """

        if energy2 < energy1:
            return True
        boltzmann_factor = np.exp(-(energy2-energy1)/temperature)
        uniform_random = np.random.random()
        return bool(boltzmann_factor > uniform_random)

    def write_initial_information(self, n_steps: int,
                                  temperature: float, conv_crit: float) -> None:
        """ Print out the initial basin-hopping information """

        self.logger.info(f"Steps = {n_steps}, Temperature = {temperature}, conv_crit = {conv_crit}")

    def write_final_information(self, start_time: float,
                                end_time: float) -> None:
        """ Print out the final basin-hopping information """

        self.logger.info(f"Located {self.ktn.n_minima} distinct minima\n")
        self.logger.info(f"Basin-hopping completed in "
                          f"{end_time - start_time}\n\n")
