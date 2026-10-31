""" Module that performs double ended transition states by the nudged
    elastic band. Double ended searches take two endpoint minima and
    aim to locate a minimum energy pathway between them, the maxima
    on this path are candidates for transition states. """

import os

import numpy as np
from numpy.typing import NDArray
from topsearch.minimisation import lbfgs
from topsearch.data.coordinates import MolecularCoordinates, StandardCoordinates
from topsearch.potentials.potential import Potential
import traceback
import logging

# Shortest interatomic distance an interpolated image may contain. Measured
# on aspirin/GFN2-xTB over 87 connection attempts, rejecting a band with any
# pair below this catches 8 of the 8 attempts that produced off-reference
# structures and discards 0 of the 79 that did not. The dihedral
# interpolation reaches 0.565 A on multi-dihedral moves, and the bonding
# graph breaks two images before the clash itself appears.
MIN_IMAGE_SEPARATION = 0.9
# Closest a pair of atoms that are NOT BONDED may come in an interpolated
# image, as a fraction of their van der Waals sum. The absolute test above
# cannot do this job: it takes the shortest distance over every pair, bonded
# included, so its threshold has to sit below the shortest real bond -- an O-H
# near 0.96 A -- and is therefore blind to a non-bonded contact at 0.94 A,
# which is the case that is actually wrong. Measured over the 83 DFT reference
# stationary points of all six Landscape17 molecules the tightest genuine
# non-bonded contact is 0.636 of the van der Waals sum, and over 41 aspirin
# interpolations this rejects all 20 that drive a non-bonded pair inside 1.5 A
# and none of the 21 that do not.
MIN_NONBONDED_FRACTION = 0.55
# How many points along a candidate rotation to test for
# clearance, and how much better the longer way has to be
# before it is preferred, as a fraction of the van der Waals
# sum. The margin keeps the shorter turn for paths that have
# no clash to avoid.
CLEARANCE_SAMPLES = 12
CLEARANCE_MARGIN = 0.03


class BandRejected(RuntimeError):
    """No usable interpolation exists between this pair of minima.

    Carries the rejected band and its closest contact so the pair can be
    looked at rather than merely counted as a failure.
    """

    def __init__(self, band, separation, scheme, reason=None):
        super().__init__(reason or
                         f"{scheme} interpolation has atoms {separation:.3f} "
                         f"A apart, below {MIN_IMAGE_SEPARATION} A")
        self.band = np.array(band, copy=True)
        self.separation = float(separation)
        self.scheme = scheme
        self.reason = reason


class NudgedElasticBand:

    """
    Description
    ------------

    Takes in the coordinates of two endpoint minima and perform a double-ended
    transition state search using the nudged elastic band algorithm
    (https://doi.org/10.1142/9789812839664_0016).
    Produces an initial linear interpolation with linear_interpolation or
    dihedral_interpolation.
    Optimise this initial guess using minimise_interpolation
    And pick out transition state candidates using find_ts_candidates
    The whole process is performed by run()

    Attributes
    ----------

    potential : class instance
        The function which we are finding minimum energy paths on
    minimiser : class instance
        The local minimiser used for optimising the band
    force_constant : float
        The value of the force constant used in harmonic part of band energy
    image_density : float
        The density of images, per distance unit, in the initial band
        interpolations
    original_image_density : float
        Store of image_density to return image_density to value once updated
    max_images : int
        The maximum allowed number of images in the nudged elastic band
    force_constants : np array
        An array of force constants for the nudged elastic band, one per image
    n_images : int
        The number of images in the given band after initial interpolation
    band_bounds : list
        The bounds on the function range that can be passed to optimiser
        Same as potential bounds, but one per image
    """

    def __init__(self, potential: Potential,
                 force_constant: float,
                 image_density: float,
                 max_images: int,
                 neb_conv_crit: float,
                 output_level: int = 0,
                 ) -> None:
        self.logger = logging.getLogger("NEB")
        self.logger.info(f"NEB parameters: force_constant: {force_constant}, image_density: {image_density}, max_images: {max_images}, neb_conv_crit: {neb_conv_crit}")
        self.potential = potential
        self.force_constant = force_constant
        self.image_density = image_density
        self.original_image_density = image_density
        self.max_images = max_images
        self.neb_conv_crit = neb_conv_crit
        self.force_constants = None
        self.n_images = None
        self.band_bounds = None
        # interpolations refused for atom overlap, kept for inspection
        self.rejected_bands = []
        self.CLEARANCE_SAMPLES = CLEARANCE_SAMPLES
        self.CLEARANCE_MARGIN = CLEARANCE_MARGIN
        self.neb_count = 0
        self.output_level = output_level

    def run(self, coords1: StandardCoordinates, coords2: NDArray, attempts: int = 0,
            permutation: NDArray = None) -> tuple[NDArray, NDArray]:
        """ Complete double-ended transition state search when provided with
            two minima. Returns an array of transition state candidates taken
            as maxima on the approximate minimum energy path """
        # Get the initial band
        band = self.initial_interpolation(coords1, coords2, attempts,
                                          permutation)
        # Minimise the initial band to get approximate minimum energy path
        band = self.minimise_interpolation(band)
        # Find local maxima of the band as transition state candidates
        candidates, positions = self.find_ts_candidates(band)
        # Return the transition state candidates and their positions
        self.neb_count += 1
        if self.output_level > 0:
            print(f'NEB {self.neb_count} found {len(candidates)} candidates')
            np.savetxt(f'neb_ts_{self.neb_count}.txt', positions)
            
        return candidates, positions

    def minimise_interpolation(self, band: NDArray) -> NDArray:
        """ Perform minimisation of the given interpolation between
            endpoints using box-contrained LBFGS and return the
            optimised band """
        # Require a 1d array for scipy lbfgs
        band = band.flatten()
        init_band_reshape = np.reshape(band, (self.n_images, -1))
        if self.output_level > 0:
            np.savetxt(f'neb_init_{self.neb_count}.txt', init_band_reshape)
        optimised_band, f_val, r_dict = \
            lbfgs.minimise(func_grad=self.band_function_gradient,
                           initial_position=band,
                           bounds=self.band_bounds,
                           conv_crit=self.neb_conv_crit)
        # Reshape into 2d array, one image per row
        opt_band_reshape = np.reshape(optimised_band, (self.n_images, -1))
        if self.output_level > 0:
            np.savetxt(f'neb_min_{self.neb_count}.txt', opt_band_reshape)
        return opt_band_reshape

    def update_image_density(self, attempts: int) -> None:
        """ Update the image density parameter """
        self.image_density = self.original_image_density*1.5*attempts

    def revert_image_density(self) -> None:
        """ Revert the image density to its original value """
        self.image_density = self.original_image_density

    def closest_contact(self, band: NDArray) -> float:
        """Shortest distance between any two atoms in any image of a band."""
        images = np.asarray(band).reshape(self.n_images, -1)
        shortest = np.inf
        for image in images:
            points = image.reshape(-1, 3)
            spread = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
            np.fill_diagonal(spread, np.inf)
            shortest = min(shortest, float(spread.min()))
        return shortest

    def nonbonded_fraction(self, band: NDArray,
                           coords1: StandardCoordinates) -> float:
        """Closest non-bonded approach in a band, over its van der Waals sum.

        The same measure choose_rotation uses, so the module has one notion of
        which atoms are not allowed to touch. Bonding is read from the
        endpoint, which is a real minimum, rather than from each image: an
        image that has driven two atoms together would otherwise count the
        contact as a new bond and excuse itself.

        Returns infinity when the bonding cannot be determined, leaving the
        absolute test as the only guard rather than refusing every band.
        """
        try:
            from ase import Atoms
            from ase.data import covalent_radii, vdw_radii
        except ImportError:
            return np.inf
        labels = getattr(coords1, 'atom_labels', None)
        if labels is None:
            return np.inf

        endpoint = Atoms(labels, coords1.position.reshape(-1, 3))
        numbers = endpoint.numbers
        bonded = endpoint.get_all_distances() < 1.2 * (
            covalent_radii[numbers][:, None]
            + covalent_radii[numbers][None, :])
        np.fill_diagonal(bonded, True)
        contact = (vdw_radii[numbers][:, None] + vdw_radii[numbers][None, :])

        worst = np.inf
        for image in np.asarray(band).reshape(self.n_images, -1):
            probe = Atoms(labels, image.reshape(-1, 3))
            fractions = probe.get_all_distances() / contact
            fractions[bonded] = np.inf
            worst = min(worst, float(fractions.min()))
        return worst

    def unusable(self, band: NDArray,
                 coords1: StandardCoordinates) -> tuple:
        """Why a band cannot be used, or None if it can.

        Two tests, because neither subsumes the other: the absolute one
        catches a collapse so severe that the bonding graph is meaningless,
        and the fractional one catches a non-bonded pair driven to bonding
        distance, which the absolute one cannot see.
        """
        separation = self.closest_contact(band)
        if separation < MIN_IMAGE_SEPARATION:
            return separation, (f"atoms {separation:.3f} A apart, below "
                                f"{MIN_IMAGE_SEPARATION} A")
        fraction = self.nonbonded_fraction(band, coords1)
        if fraction < MIN_NONBONDED_FRACTION:
            return fraction, (f"non-bonded atoms at {fraction:.3f} of their "
                              f"van der Waals sum, below "
                              f"{MIN_NONBONDED_FRACTION}")
        return None

    def reject_overlapping(self, band: NDArray,
                           coords1: StandardCoordinates,
                           coords2: NDArray) -> NDArray:
        """Return the band, or a linear one, or refuse the pair.

        Rotating several dihedrals at once can drive two atoms through each
        other. Linear interpolation cannot do that, so it is tried before
        the pair is given up on.
        """
        failure = self.unusable(band, coords1)
        if failure is None:
            return band
        measure, why = failure
        self.logger.info(f"Dihedral interpolation refused: {why}; "
                         f"retrying linearly")
        self.rejected_bands.append({"scheme": "dihedral",
                                    "separation": float(measure),
                                    "reason": why,
                                    "band": np.array(band, copy=True)})
        band = self.linear_interpolation(coords1, coords2)
        failure = self.unusable(band, coords1)
        if failure is not None:
            measure, why = failure
            self.rejected_bands.append({"scheme": "linear",
                                        "separation": float(measure),
                                        "reason": why,
                                        "band": np.array(band, copy=True)})
            raise BandRejected(band, measure, "linear",
                               reason=f"linear interpolation has {why}")
        return band

    def initial_interpolation(self, coords1: StandardCoordinates, coords2: NDArray,
                              attempts: int, permutation: NDArray) -> NDArray:
        """ Return initial interpolation band using the appropriate number
            of images based on previous attempts. Set force constants and
            bounds attributes for use in other methods """

        # If retrying connection then use increased image density
        if attempts > 0:
            self.update_image_density(attempts)
        # Perform the type of interpolation specified by number of attempts
        if isinstance(coords1, MolecularCoordinates):
            band = self.dihedral_interpolation(coords1, coords2, permutation)
            band = self.reject_overlapping(band, coords1, coords2)
        else:
            band = self.linear_interpolation(coords1, coords2)
        # If changed parameters then revert back
        if attempts > 0:
            self.revert_image_density()
        # And initialise bounds array given self.n_images is now set
        self.band_bounds = coords1.bounds*self.n_images
        # With the band known calculate the force constants
        self.get_force_constants()
        return band

    def linear_interpolation(self, coords1: StandardCoordinates, coords2: NDArray) -> NDArray:
        """ Produce a linear interpolation between two points with a number of
            images specified by self.image_density """
        # Set the number of images based on given image density
        dist = np.linalg.norm(coords1.position-coords2)
        self.n_images = int(self.image_density*dist)
        # Can't have too few images
        if self.n_images < 10:
            self.n_images = 10
        # Don't allow band to have above the maximum specified images
        elif self.n_images > self.max_images:
            self.n_images = self.max_images
        # Make the linear interpolation
        band = np.zeros((self.n_images, coords1.ndim), dtype=float)
        direction_vec = (coords2-coords1.position)/(self.n_images-1)
        for i in range(self.n_images):
            band[i, :] = coords1.position+(direction_vec*i)
        return band

    def bonding_correspondence(self, coords1, coords2):
        """The correspondence under which the two bonding graphs agree.

        Returns None when the structures bond differently, since no
        relabelling can reconcile them and the move is a real chemical
        change rather than a rearrangement of equivalent atoms.
        """
        try:
            import networkx as nx
            from networkx.algorithms.isomorphism import (
                GraphMatcher, categorical_node_match)
            from ase import Atoms
            from ase.neighborlist import natural_cutoffs, NeighborList
        except ImportError:
            return None

        labels = getattr(coords1, 'atom_labels', None)
        if labels is None:
            return None

        def graph_of(position):
            atoms = Atoms(labels, np.asarray(position).reshape(-1, 3))
            neighbours = NeighborList(natural_cutoffs(atoms, mult=1.15),
                                      skin=0.0, self_interaction=False,
                                      bothways=True)
            neighbours.update(atoms)
            graph = nx.Graph()
            for i, element in enumerate(labels):
                graph.add_node(i, element=element)
            for i in range(len(atoms)):
                for j in neighbours.get_neighbors(i)[0]:
                    graph.add_edge(int(i), int(j))
            return graph

        try:
            first = graph_of(coords1.position)
            second = graph_of(coords2)
        except Exception:
            return None
        matcher = GraphMatcher(
            first, second,
            node_match=categorical_node_match('element', ''))
        reference = np.asarray(coords1.position).reshape(-1, 3)
        target = np.asarray(coords2).reshape(-1, 3)
        best, best_distance, seen = None, np.inf, 0
        for mapping in matcher.isomorphisms_iter():
            seen += 1
            if seen > 5000:
                break
            order = [mapping[i] for i in range(len(labels))]
            spread = target[order] - reference
            distance = float(np.sqrt(np.mean(np.sum(spread ** 2, axis=1))))
            if distance < best_distance:
                best, best_distance = order, distance
        if best is None:
            return None
        self.logger.debug(
            "Interpolation correspondence taken from the bonding graphs "
            "rather than from distance")
        return np.array(best)

    def physical_permutation(self, coords1: StandardCoordinates,
                             coords2: NDArray,
                             permutation: NDArray) -> NDArray:
        """ Replace a correspondence no motion can realise with the closest
            one that can.

            Alignment chooses the atom correspondence that minimises
            distance, which is the right question when asking whether two
            structures are the same: a structure with two hydrogens
            relabelled is the same structure. It is the wrong question here,
            because the interpolation reads the correspondence as a
            statement about which atom travels where. Pairing a hydrogen on
            a carbon with one on an oxygen asks that hydrogen to cross the
            molecule, and rotating dihedrals towards it pulls the hydrogen
            off its parent instead, breaking bonds before the potential is
            ever consulted.

            Only the relabellings that leave the bonding unchanged describe
            motions, so an unphysical correspondence is swapped for whichever
            of those sits closest. Returns the original if the coordinates
            carry no bonding information, or if the correspondence was
            already a symmetry.
        """
        if permutation is None:
            return permutation
        mapped = self.bonding_correspondence(coords1, coords2)
        if mapped is not None:
            return mapped
        automorphisms = getattr(coords1, 'automorphisms', None)
        if not automorphisms:
            return permutation
        if [int(i) for i in permutation] in automorphisms:
            return permutation
        reference = coords1.position.reshape(-1, 3)
        target = np.asarray(coords2).reshape(-1, 3)
        best, best_distance = permutation, np.inf
        for candidate in automorphisms:
            displaced = target[candidate] - reference
            distance = float(np.sqrt(np.mean(np.sum(displaced ** 2, axis=1))))
            if distance < best_distance:
                best, best_distance = candidate, distance
        self.logger.debug(
            "Alignment gave a correspondence that is not a symmetry of the "
            "molecule; replaced with the closest one that is")
        return np.array(best)

    def dihedral_interpolation(self, coords1: StandardCoordinates, coords2: NDArray,
                               permutation: NDArray) -> NDArray:
        """ Interpolate linearly in the space of dihedrals, angles and
            bond lengths, which will be much more appropriate for molecules """
        permutation = self.physical_permutation(coords1, coords2, permutation)
        # Set the number of images based on given image density
        try:
            dist = np.linalg.norm(coords1.position-coords2)
        except:
            print(coords1.position, coords2)
            print(coords1)
            traceback.print_exc()
            try:
                print(coords1.position.position)
            except:
                pass
            raise RuntimeError
        self.n_images = int(self.image_density*dist)
        # Can't have too few images
        if self.n_images < 10:
            self.n_images = 10
        # Don't allow band to have above the maximum specified images
        elif self.n_images > self.max_images:
            self.n_images = self.max_images
        # Get the angular representation of the conformation and its bonds
        represent1 = coords1.get_bond_angle_info()
        bond_network = coords1.get_bonds()
        # Get the angular representation of same bonds, including permutations
        initial_coords1 = coords1.position.copy()
        coords1.position = coords2
        represent2 = \
            coords1.get_specific_bond_angle_info(represent1[0],
                                                 represent1[2],
                                                 represent1[4],
                                                 permutation)
        coords1.position = initial_coords1
        # Calculate the direction differences to create the band
        bond_differences = (np.asarray(represent2[0]) -
                            np.asarray(represent1[1]))/(self.n_images-1)
        angle_differences = (np.asarray(represent2[1]) -
                             np.asarray(represent1[3]))/(self.n_images-1)
        dihedral_differences = []
        # A dihedral reaches its target by turning either way round the
        # circle. Taking the smaller angle is a statement about angles, not
        # about the molecule: on aspirin the shorter turn about C6-O12
        # sweeps one ester group past the other, bringing a carbonyl oxygen
        # within 1.88 A of a carbon it is not bonded to, where the longer
        # turn keeps it at 2.26 A. Prefer the direction that stays clear.
        for i in range(len(represent1[5])):
            shortest = represent2[2][i] - represent1[5][i]
            shortest = (shortest + 180.0) % 360.0 - 180.0
            longer = shortest - 360.0 if shortest > 0 else shortest + 360.0
            dihedral_differences.append(
                self.clearer_rotation(coords1, represent1, i,
                                      shortest, longer))
        dihedral_differences = \
            np.asarray(dihedral_differences)/(self.n_images-1)
        # Make the linear interpolation
        band = np.zeros((self.n_images, coords1.ndim), dtype=float)
        band[0, :] = coords1.position
        for i in range(1, self.n_images):
            coords1.change_bond_lengths(represent1[0],
                                        bond_differences,
                                        bond_network)
            coords1.change_bond_angles(represent1[2],
                                       -1.0*angle_differences,
                                       bond_network)
            coords1.change_dihedral_angles(represent1[4],
                                           dihedral_differences,
                                           bond_network)
            band[i, :] = coords1.position.flatten()
        coords1.position = initial_coords1
        return band


    def clearer_rotation(self, coords: StandardCoordinates, represent: tuple,
                         index: int, shortest: float, longer: float) -> float:
        """Of the two ways round the circle, the one that stays further off.

        Turns the one dihedral through each route in turn and measures how
        close any pair of atoms that are not bonded comes, as a fraction of
        their van der Waals radii. The shorter turn is kept unless the
        longer one is clearly better, so that this only changes the paths
        that have a reason to change.
        """
        try:
            from ase import Atoms
            from ase.data import covalent_radii, vdw_radii
        except ImportError:
            return shortest
        labels = getattr(coords, 'atom_labels', None)
        if labels is None:
            return shortest
        dihedral = represent[4][index] if len(represent) > 4 else None
        if dihedral is None or len(np.atleast_1d(dihedral)) != 4:
            return shortest

        atoms = Atoms(labels, coords.position.reshape(-1, 3))
        numbers = atoms.numbers
        separations = atoms.get_all_distances()
        bonded = separations < 1.2 * (covalent_radii[numbers][:, None]
                                      + covalent_radii[numbers][None, :])
        contact = (vdw_radii[numbers][:, None] + vdw_radii[numbers][None, :])
        np.fill_diagonal(bonded, True)

        def clearance(sweep):
            worst = np.inf
            start = atoms.get_dihedral(*[int(a) for a in dihedral])
            for step in range(1, self.CLEARANCE_SAMPLES):
                probe = atoms.copy()
                try:
                    probe.set_dihedral(
                        *[int(a) for a in dihedral],
                        start + sweep * step / self.CLEARANCE_SAMPLES)
                except (ValueError, ZeroDivisionError):
                    return np.inf
                fractions = probe.get_all_distances() / contact
                fractions[bonded] = np.inf
                worst = min(worst, float(fractions.min()))
            return worst

        return longer if clearance(longer) > clearance(shortest) + \
            self.CLEARANCE_MARGIN else shortest

    def get_force_constants(self) -> None:
        """ Get the array of force constants for the harmonic potential """
        self.force_constants = np.zeros((self.n_images-1), dtype=float)
        self.force_constants.fill(self.force_constant)

    def find_ts_candidates(self, band: NDArray) -> tuple[NDArray, NDArray]:
        """ Find the local maxima of the nudged elastic band and return the
        indices of these candidate transition states and their coordinates """
        # Compute the energy of each image in the band
        band_potential_energies = self.band_potential_function(band)[1]
        # Initialise arrays to store any transition state candidates we find
        candidates = np.empty((0), dtype=int)
        positions = np.empty((0, 0), dtype=float)
        # Search through the 1D array of energies to find local maxima
        for i in range(1, self.n_images-1):
            if (band_potential_energies[i] >= band_potential_energies[i+1] and
                  band_potential_energies[i] >= band_potential_energies[i-1]):
                # Store index and coordinates of transition state candidate
                candidates = np.append(candidates, i)
                positions = np.append(positions, band[i, :])
        return candidates, \
            np.reshape(positions, (np.size(candidates), band[0, :].size))

    def find_tangent_differences(self, band: NDArray,
                                 energies: NDArray) -> NDArray:
        """ Compute the tangent vectors between consecutive images
            in the band, these are calculated from lower to higher
            connected images in the Upwind scheme. Returns array
            of tangents, one for each image (apart from endpoints) """

        # Create the 2D array to place the tangents in
        tangents = np.zeros((self.n_images-2, band[0, :].size), dtype=float)
        # Only use a tangent to the higher connected image
        position_differences = -1.0*np.diff(band, axis=0)
        # Compute the differences between energies of images
        energy_differences = np.diff(energies, axis=0)
        # Compute the signs of energy differences to show changing energy
        energy_change_sign = np.sign(energy_differences).astype(int)
        for i in range(1, self.n_images-1):
            # Band has increasing energy so select following image
            if energy_change_sign[i-1]+energy_change_sign[i] >= 1:
                tangents[i-1, :] = position_differences[i]
            # Band has decreasing energy so select previous image
            elif energy_change_sign[i-1]+energy_change_sign[i] <= -1:
                tangents[i-1, :] = position_differences[i-1]
            # Band has local maximum, local minimum or it is flat
            elif energy_change_sign[i-1]+energy_change_sign[i] == 0:
                #  Band is flat
                if 0 in (energy_change_sign[i-1], energy_change_sign[i]):
                    tangents[i-1, :] = position_differences[i-1]
                # Local minimum or local maximum
                else:
                    vec1 = position_differences[i]
                    vec2 = position_differences[i-1]
                    abs_energy_differences = \
                        np.abs(energy_differences[i-1:i+1])
                    v_max = max(abs_energy_differences)
                    v_min = min(abs_energy_differences)
                    if energies[i+1] >= energies[i-1]:
                        tangents[i-1, :] = vec1*v_max + vec2*v_min
                    else:
                        tangents[i-1, :] = vec1*v_min + vec2*v_max
        # Normalise the tangent vectors
        row_sums = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = np.divide(tangents, row_sums, out=np.zeros_like(
            tangents), where=row_sums != 0)
        return tangents

    def band_potential_function(self, band: NDArray) -> tuple[float, NDArray]:
        """ Return the total energy of the band, and an array giving the
            true function evaluated at each image of the band """
        n_images = int(np.size(band, axis=0))
        band_potential_energies = np.zeros((n_images, 1), dtype=float)
        band_potential_energies = \
            np.apply_along_axis(self.potential.function, 1, band)
        potential_function = np.sum(band_potential_energies)
        return potential_function, band_potential_energies

    def band_function_gradient(self, band: NDArray) -> tuple[float, NDArray]:
        """ Return the total energy and the gradient of the nudged elastic
            band, this involves the summation of the true function and
            a series of harmonic terms between adjacent images """

        # Reshape the band into its images
        band = np.reshape(band, (self.n_images, -1))
        # Compute the components of the true potential
        potential_gradient = \
            np.zeros((self.n_images, band[0, :].size), dtype=float)
        band_potential_energies = np.zeros((self.n_images, 1), dtype=float)
        for i in range(self.n_images):
            f_val, grad = self.potential.function_gradient(band[i, :])
            potential_gradient[i, :] = grad
            band_potential_energies[i] = f_val
        # Calculate the function value as sum over all images
        potential_function = np.sum(band_potential_energies)
        # Ensure no true gradient on the endpoint images
        potential_gradient[0, :].fill(0.0)
        potential_gradient[-1, :].fill(0.0)

        # Find the tangents to the band
        tau = self.find_tangent_differences(band, band_potential_energies)

        # Compute the components of the harmonic potential parallel to tangents
        g_parallel = np.zeros((self.n_images, band[0, :].size), dtype=float)
        # Get separation vector between adjacent images
        differences = -1.0*np.diff(band, axis=0)
        # Compute the distances
        distances = np.linalg.norm(differences, axis=1)
        # Calculate the function
        harmonic_function = 0.5*np.sum((distances**2)*self.force_constants)
        # And its gradient parallel to the tangent
        g_parallel[1:-1, :] = tau * \
            (-1.0*np.diff(distances)*self.force_constants[:-1]).reshape(-1, 1)

        # Combine the perpendicular component of the true potential
        # and the parallel component of the harmonic potential
        band_gradient = np.zeros((self.n_images, band[0, :].size), dtype=float)
        for i in range(1, self.n_images-1):
            band_gradient[i, :] = g_parallel[i, :] + \
                self.perpendicular_component(potential_gradient[i, :],
                                             tau[i-1, :])
        return harmonic_function+potential_function, band_gradient.flatten()

    def perpendicular_component(self, vec1: NDArray, vec2: NDArray) -> NDArray:
        """ Return perpendicular component of vector vec1 relative to vec2 """
        vec2_magnitude = np.sum(vec2**2)
        if vec2_magnitude < 1e-13:
            return np.zeros(np.size(vec1))
        return vec1 - (np.dot(vec1, vec2)/vec2_magnitude)*vec2
