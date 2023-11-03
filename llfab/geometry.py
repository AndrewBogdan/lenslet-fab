"""Tools for geometric manipulation and creation of the lenslet geometry."""

# --- Libraries ---
from typing import List, Optional, Tuple, TypeAlias, Union
import collections
import copy
import functools
import logging
import math
import numbers
import random
import time

from typing import TypeVar
from collections.abc import Collection, Sequence
from numbers import Real

from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv
import scipy

from llfab import geodesic, harness, toolpaths, util
from llfab.harness import Inst as In


# Type Aliases
SphPoint = TypeVar('SphPoint')
XYZPoint = Tuple[float, float, float]
PlotArgs: TypeAlias = Union[bool, dict]
Selection: TypeAlias = Union[None, Sequence[int], Sequence[bool], str]

# Logger
_logger = logging.getLogger(__name__)


# --- Classes -----------------------------------------------------------------
class Radian(float, numbers.Real):
    """A Radian class which kees radians in the range +pi, -pi."""
    def __new__(cls, value=0):
        value_mod = value % (2 * math.pi)
        value_rad = min([value_mod, value_mod - (2 * math.pi)], key=abs)
        return super().__new__(cls, value_rad)

    # def __abs__(self):
    #     return min(float.__abs__(self), float.__abs__(2*np.pi - self))


class SphPoint(collections.namedtuple('_SphPoint', ['incline', 'azimuth'])):
    """A point on a unit sphere."""

    def __new__(cls, incline: Real = 0.0, azimuth: Real = 0, *args, **kwargs):
        return super().__new__(cls, Radian(incline) % np.pi, Radian(azimuth))

    # --- Constructors ---
    @classmethod
    def from_xyz(cls, x: Real, y: Real, z: Real, *args, **kwargs):
        magnitude = np.sqrt(x ** 2.0 + y ** 2.0 + z ** 2.0)
        if magnitude == 0: raise ValueError('Magnitude cannot be zero.')
        incline = np.arccos(z / magnitude)
        azimuth = np.angle(x + 1j * y)
        return cls(incline, azimuth, *args, **kwargs)

    # --- Properties ---
    @property
    def riemann(self):
        """The value of the point as projected onto the Riemann sphere."""
        return np.tan(self.incline / 2) * np.exp(1j * self.azimuth)

    @property
    def x(self):
        return np.sin(self.incline) * np.cos(self.azimuth)

    @property
    def y(self):
        return np.sin(self.incline) * np.sin(self.azimuth)

    @property
    def z(self):
        return np.cos(self.incline)

    @property
    def xyz(self):
        return self.x, self.y, self.z

    # --- Math ---
    def arc_to(self, other: SphPoint) -> Radian:
        """Calculates the arc length to another SphPoint."""
        return Radian(np.arccos(np.dot(self.xyz, other.xyz)))

    def axis_to(self, other: SphPoint) -> SphPoint:
        """Calculates the axis around which you would rotate the
        unit sphere to go from self to other.

        You can also think of it as the vector normal to the plane
        which contains self, other and the origin.
        """
        return SphPoint.from_xyz(*np.cross(self.xyz, other.xyz))

    cross = axis_to

    def heading_to(self, other: SphPoint) -> SphPoint:
        """Finds the heading to a point, where the heading to North (0, 0, 1)
        is always zero. If self is North, then it returns the azimuth of other.

        The idea of this is that, if you consider self as the North pole, the
        heading is always the azimuth to other.

        That is, SphPoint(incline=A.arc_to(B), azimuth=A.heading_to(B))
        should be equal to B.with_pole(A).
        """
        return other.with_pole(self).azimuth

    def with_pole(self, pole: SphPoint) -> SphPoint:
        """Calculates the coordinates of this point considering pole
        as the North pole."""
        if pole.incline == 0:
            return self
        true_north = SphPoint()
        axis = true_north.axis_to(pole)
        north = true_north.rotate_about(axis, -pole.incline)
        new_self = self.rotate_about(axis, -pole.incline)
        new_self = new_self.rotate_about(true_north, -north.azimuth)
        return new_self

    def rotate_about(self, axis: SphPoint, arc: Radian) -> SphPoint:
        """Rotates self around axis by arc, so that, if axis is
        on the equator opposite to self, the cross product of
        self and the new point is axis * sin(arc)."""
        cross_matrix = np.array([
            [0, -axis.z, axis.y],
            [axis.z, 0, -axis.x],
            [-axis.y, axis.x, 0],
        ])

        axis_vector = np.asarray([axis.xyz])
        outer_product = axis_vector.T @ axis_vector

        rotation_matrix = (
                np.cos(arc) * np.identity(3) +
                np.sin(arc) * cross_matrix +
                (1 - np.cos(arc)) * outer_product
        )

        return SphPoint.from_xyz(
            *(rotation_matrix @ self.xyz)
        )

    # --- Magic ---
    def __repr__(self):
        return f'{self.__class__.__name__}' \
               f'(incline={self.incline:.3f}, azimuth={self.azimuth:.3f})'


class LaseGeometry(Sequence):
    """A class which represents a collection of lases, with utilities for editing and optimizing the geometry.
    It is a sequence, but the order does not represent toolpath order."""

    def __init__(self, lens_diameter: float = 2.0):
        """Initialize a geometry."""
        self.lens_diameter = lens_diameter

        self.lases = None
        self.edges = None
        self.headings = None
        self.corners = None
        self.flags = {}

        self._n_gons = {}

    def clear_caches(self, kind='all'):
        """Clear the caches, optionally picking a subset of caches to clear.
        You can provide one of or a list of:
        - 'all': Clear all caches.
        - 'graph': Clear the caches related to the graph representation of the
            geometry. Clear this when you add or remove lases.
        - 'vector': Clear the caches related to the actual position of lases on
            the sphere. Clear these if you change the location of a lase or
            rotate the geometry."""
        if not isinstance(kind, Collection): kind = (kind, )
        _logger.debug(f'Clearing {kind} caches...')
        if 'all' in kind or 'vector' in kind or 'graph' in kind:
            self.sph.cache_clear()
            self.get_edge_distances.cache_clear()
            self.get_lase_distances.cache_clear()
            self.get_min_lase_distances.cache_clear()
            self.get_mean_lase_distances.cache_clear()
        if 'all' in kind or 'graph' in kind:
            self.edges_of.cache_clear()
            self.neighborhood_of.cache_clear()
            self.neighbors_of.cache_clear()
            self.degree_of.cache_clear()
            self.n_gons.cache_clear()
            self.get_edge_indices_map.cache_clear()

    # --- Utility -------------------------------------------------------------
    # ------ Marking & Masking ------------------------------------------------
    def add_flag(self,
                 flag: str,
                 mask: Optional[Sequence[bool]] = None,
                 indices: Optional[Sequence[int]] = None):
        """Marks the given lases with the given flag. Supply exactly one of
        mask or indices.

        Args:
            flag: The name of the flag. This can be used like lg['flag'].
            mask: A list of booleans of the same length as the geometry. This
                will add just the True lases to the flag.
            indices: The indices of lases to flag.
        """
        if mask is not None == indices is not None:
            raise TypeError('Supply exactly one of mask or indices.')

        if mask is not None:
            indices = np.asarray(range(len(self)))[mask]
        self.flags[flag] = np.array(sorted(indices), dtype=int)

    def mask(self, selection: Selection = None) -> np.ndarray[bool]:
        """Returns a mask corresponding to the selection.
        If you supply:
        - None: you get all indices.
        - A list of booleans (a mask): you get your input, fed back to you.
        - A list of ints: you get the mask corresponding to those indices.
        - A string: you get the mask corresponding to that flag."""
        mask = np.zeros(len(self), dtype=bool)
        if selection is None:
            return np.ones(len(self), dtype=bool)
        elif isinstance(selection, str):
            mask[self.flags[selection]] = True
        elif isinstance(selection, Collection):
            selection = np.asarray(list(selection))
            if selection.dtype == np.dtype('bool'):
                return selection
            else:
                mask[selection] = True
        else:
            raise TypeError(f'Invalid type {type(selection)}')

        return mask

    def select(self, selection: Selection = None) -> np.ndarray[int]:
        """Returns the indices of the lases corresponding to the selection.
        If you supply:
        - None: you get all indices.
        - A list of booleans (a mask): you get the indices corresponding to
            that mask.
        - A list of ints: you get your input, fed back to you, sorted.
        - A string: you get the value of that flag."""
        if selection is None:
            return np.array(range(len(self)))
        elif isinstance(selection, str):
            return self.flags[selection]
        elif isinstance(selection, Collection):
            selection = np.asarray(list(selection))
            if selection.dtype == np.dtype('bool'):
                return np.array(range(len(self)))[selection]
            else:
                return selection
        else:
            raise TypeError(f'Invalid type {type(selection)}')

    # ------ Graph theoretic tools --------------------------------------------
    @functools.lru_cache(maxsize=None)
    def edges_of(self, lase_index: int):
        """Get all the edges_of associated with a lase."""
        return np.array([edge for edge in self.edges if lase_index in edge])

    @functools.lru_cache(maxsize=None)
    def neighborhood_of(self, lase_index: int, distance: int = 1):
        """Get all the neighbors_of of a lase that are within distance."""
        # First we handle the base cases
        if distance == 0:
            return {lase_index}
        elif distance == 1:
            return set(self.edges_of(lase_index).ravel())
        else:
            # Recursive step: get the most distant neighbors_of, and
            #  find their neighbors_of, then add those to the known neighborhood_of.
            distant_neighbors = self.neighbors_of(
                lase_index=lase_index,
                distance=distance - 1,
            )
            neighborhood = self.neighborhood_of(
                lase_index=lase_index,
                distance=distance - 1,
            )
            return set.union(neighborhood, *(
                self.neighborhood_of(lase_index=neighbor, distance=1, )
                for neighbor in distant_neighbors
            ))

    @functools.lru_cache(maxsize=None)
    def neighbors_of(self, lase_index: int, distance: int = 1):
        """Get all the neighbors_of of a lase that are exactly distance away."""
        # First we handle the base cases
        if distance == 0:
            return {lase_index}
        elif distance == 1:
            return self.neighborhood_of(lase_index=lase_index) - {lase_index}
        else:
            # Recursive step: rely on self.neighborhood_of and caching.
            bigger_neighborhood = self.neighborhood_of(
                lase_index=lase_index,
                distance=distance,
            )
            smaller_neighborhood = self.neighborhood_of(
                lase_index=lase_index,
                distance=distance - 1,
            )
            return bigger_neighborhood - smaller_neighborhood

    @functools.lru_cache(maxsize=None)
    def degree_of(self, lase_index: int):
        """Get the number of neighbors_of of a lase."""
        return len(self.neighbors_of(lase_index))

    @functools.lru_cache(maxsize=None)
    def n_gons(self, num_sides: int):
        if num_sides in self._n_gons:
            return self._n_gons[num_sides]
        else:
            return np.array([ind for ind in range(len(self.lases))
                             if self.degree_of(ind) == num_sides])

    # --- Properties ----------------------------------------------------------
    @property
    def inclines(self):
        return np.arccos(self.lases[:, 2])

    @property
    def azimuths(self):
        return np.angle(self.lases[:, 0] + 1j * self.lases[:, 1])

    @property
    def pentagons(self):
        return self.n_gons(5)

    @pentagons.setter
    def pentagons(self, indices):
        self._n_gons[5] = indices

    @property
    def hexagons(self):
        return self.n_gons(6)

    @hexagons.setter
    def hexagons(self, indices):
        self._n_gons[6] = indices

    # --- Simple Math Help ----------------------------------------------------
    def geodesic_to_chord(self, geodesic: float):
        return self.lens_diameter * np.sin(geodesic / self.lens_diameter)

    def chord_to_geodesic(self, chord: float):
        return self.lens_diameter * np.arcsin(chord / self.lens_diameter)

    def geodesic_to_euclid(self, geodesic: float):
        return 2 * np.sin(geodesic / self.lens_diameter)

    def euclid_to_geodesic(self, euclid: float):
        return self.lens_diameter * np.arcsin(euclid / 2)

    def arc_to_chord(self, arc: Radian):
        return self.lens_diameter * np.sin(arc / 2)

    def chord_to_arc(self, chord: float):
        return 2 * np.arcsin(chord / self.lens_diameter)

    def arc_to_geodesic(self, arc: Radian):
        return arc * self.lens_diameter / 2

    def geodesic_to_arc(self, geodesic: float):
        return 2 * geodesic / self.lens_diameter

    # --- Convenience Calculations --------------------------------------------
    @functools.lru_cache(maxsize=1)
    def get_edge_indices_map(self):
        """Make the lookup table that maps
        edge_lookup[left index][right index] -> index of that edge."""
        assert self.edges is not None, 'self.edges doesn\'t exist yet!'

        edge_lookup = collections.defaultdict(dict)
        for edge_idx, (left_idx, right_idx) in enumerate(self.edges):
            edge_lookup[left_idx][right_idx] = edge_idx
            edge_lookup[right_idx][left_idx] = edge_idx

        return dict(edge_lookup)

    @functools.lru_cache(maxsize=1)
    def get_edge_distances(self):
        """For each edge between lases, get the center-to-center geodesic
        distance between the two lases. Returns an array such that the edge
        self.edges[idx] will correspond to self.get_edge_distances()[idx]."""
        edge_arcs = np.asarray(
            [self.sph(start).arc_to(self.sph(end)) for start, end in
             self.edges])
        edge_dists = self.arc_to_geodesic(edge_arcs)
        return edge_dists

    @functools.lru_cache(maxsize=1)
    def get_lase_distances(self):
        """For each lase, get the distances between it and all its neighbors.
        This will be a ragged list, so don't try to put it into numpy. Returns
        a list such that self.lases[idx] will correspond to
        self.get_lase_distances()[idx], which is itself an array."""
        edge_dists = self.get_edge_distances()
        ei_map = self.get_edge_indices_map()

        dists_of_lase = []
        for index in range(len(self)):
            neighbors = self.neighbors_of(index)
            edges = [ei_map[index][neigh] for neigh in neighbors]
            dists_of_lase.append(edge_dists[edges])
        return dists_of_lase

    @functools.lru_cache(maxsize=1)
    def get_min_lase_distances(self):
        """Get the center-to-center distance of each lase to its nearest
        neighbor. Returns an array such that self.lases[idx] will be distance
        self.get_min_lase_distances()[idx] from its nearest neighbor,
        measured geodesically, center to center."""
        dists_of_lase = self.get_lase_distances()
        return np.asarray([min(dists) for dists in dists_of_lase])

    @functools.lru_cache(maxsize=1)
    def get_mean_lase_distances(self):
        """Get the average center-to-center distance of each lase to its
        neighbors. Returns an array such that self.lases[idx] will, on average,
        be distance self.get_mean_lase_distances()[idx] from its neighbors,
        measured geodesically, center to center."""
        dists_of_lase = self.get_lase_distances()
        return np.asarray([float(np.mean(dists)) for dists in dists_of_lase])

    @functools.lru_cache(maxsize=None)
    def sph(self, index):
        return SphPoint.from_xyz(*self.lases[index])

    # --- Volatile Geometric Calculations -------------------------------------

    def make_lases_geodesic(self, a_val, b_val):
        """Make a LaseGeomtry where the lases are at the vertices of a geodesic sphere."""
        _logger.info(f'Making Geodesic Sphere GS({a_val}, {b_val}).')
        lases, edges, _ = _make_geodesic_sphere(a_val, b_val, repeat=1)
        self.lases = lases
        self.edges = edges

    def set_zenith(self, coords):
        """Rotate the geometry so the zenith is at a specific location."""
        _logger.warning(f'Using inefficient method '
                        f'{self.__class__.__name__}.set_zenith.')
        zenith_sph = SphPoint.from_xyz(*coords)
        for lase_idx in range(len(self.lases)):
            lase_sph = self.sph(lase_idx)
            self.lases[lase_idx] = lase_sph.with_pole(zenith_sph).xyz

        _logger.warning(f'Zenith reset, headings and corners are now undefined.')
        self.headings = None
        self.corners = None
        self.clear_caches('vector')

    @util.depreciate
    def slice_in_half(self, cutoff=np.radians(85), drop_pentagons=True):
        """Slice the geometry in half, keeping the top half."""
        _logger.warning(f'Using inefficient method '
                        f'{self.__class__.__name__}.slice_in_half.')

        num_lases = len(self.lases)

        cutoff_mask = self.inclines < cutoff
        penta_mask = np.array([True] * num_lases)
        if drop_pentagons: penta_mask[self.pentagons] = False

        mask = cutoff_mask & penta_mask
        mask_indices = np.arange(num_lases)[mask]

        lookup = lambda idx: np.where(mask_indices == idx)[0][0]

        self.lases = self.lases[mask_indices]
        self.edges = np.array([
            (lookup(start), lookup(end)) for start, end in self.edges
            if start in mask_indices and end in mask_indices
        ])
        _logger.warning(f'Changed lase indexing, headings and corners are '
                        f'now undefined.')
        self.headings = None
        self.corners = None
        self.pentagons = list(range(np.count_nonzero(mask[:12])))
        self.hexagons = list(set(range(len(self.lases))) - set(self.pentagons))
        self.clear_caches('all')
        _logger.info('Saving pentagons and hexagons.')

    # --- Stateful Geometric Calculations -------------------------------------
    # ------ Headings ---------------------------------------------------------
    def make_headings_from_edges(self):
        """Picks headings so that they point towards an edge."""
        _logger.warning(f'Using inefficient method '
                        f'{self.__class__.__name__}.make_headings_from_edges:')
        if self.lases is None: raise ValueError('Lases undefined!')

        headings = []
        for lase_index, lase_xyz in enumerate(self.lases):
            if lase_index % min(1000, len(self.lases) // 5 + 1) == 0:
                _logger.info(f'\t{100 * lase_index / len(self.lases):.2f}% complete...')

            lase = self.sph(lase_index)
            lase_headings = []

            # Get the headings from each neighbor
            for neighbor_index in self.neighbors_of(lase_index):
                neighbor = self.sph(neighbor_index)
                lase_headings.append(lase.heading_to(neighbor))

            # Pick the one that's closest to North
            min_heading = min(lase_headings, key=abs)
            headings.append(min_heading)
        self.headings = np.array(headings)

    def minimize_headings(self):
        """Minimize the headings for the N axis to reach them, relies on
        self.n_gons correctly knowing which lases are what."""
        _logger.info('Minimizing headings...')
        minimized_headings = []
        for lase_idx, heading in enumerate(self.headings):
            poly_angle = 2 * np.pi / self.degree_of(lase_idx)
            min_heading = min((heading % poly_angle, heading % -poly_angle), key=abs)
            minimized_headings.append(min_heading)
        self.headings = np.array(minimized_headings)

    def bound_headings(self, bounds) -> np.ndarray[bool]:
        """Makes sure that abs(headings) is within bounds, and changes it
        if necessary. Returns a mask of the lases which were bounded."""
        mask = abs(self.headings) > bounds
        self.headings[mask] = bounds[mask] * np.sign(self.headings[mask])
        return mask

    # ------ Corners ----------------------------------------------------------
    def make_corners_hexagonal(self, hex_diameter_long: float):
        """Make the corners, assuming regular hexagonal lases."""
        time_start = time.time()
        _logger.debug(f'Making hexagonal corners...')
        if self.lases is None: raise ValueError('Lases undefined!')
        if self.headings is None: raise ValueError('Headings undefined!')

        arc_radius = np.arcsin(hex_diameter_long / self.lens_diameter)
        self.corners = _make_corners_polygonal(
            num_sides=6,
            lases=self.lases,
            headings=self.headings,
            arc_radius=arc_radius
        )

        _logger.debug(f'Done! Time elapsed: '
                      f'{(time.time() - time_start) * 1e3:.2f} ms')

    def make_corners_pentagonal(self, penta_diameter_long: float):
        """Make the corners, assuming regular pentagonal lases."""
        time_start = time.time()
        _logger.debug(f'Making pentagon corners...')
        if self.lases is None: raise ValueError('Lases undefined!')
        if self.headings is None: raise ValueError('Headings undefined!')

        arc_radius = np.arcsin(penta_diameter_long / self.lens_diameter)
        self.corners = _make_corners_polygonal(
            num_sides=5,
            lases=self.lases,
            headings=self.headings,
            arc_radius=arc_radius
        )

        _logger.debug(f'Done! Time elapsed: '
                      f'{(time.time() - time_start) * 1e3:.2f} ms')

    # ------ Other ------------------------------------------------------------
    def flag_in_half(self, cutoff=np.radians(80)):
        """Slice the geometry in half, keeping the top half. More specifically,
        this marks everything above the cutoff as with the flag 'upper', and
        everything below it with the flag 'lower'."""
        mask = self.inclines < cutoff
        self.add_flag('upper', mask)
        self.add_flag('lower', ~mask)

    # --- Stateless Geometric Calculations / Optimization Routines ------------
    def project_sph_to_xyv(self) -> np.ndarray:
        """Performs a azithumal equidistant projection from the zenith, turning
        a spherical geometry into a planar one. Returns a list of XY
        coordinates to lase at, of shape (N, 2)."""
        complex_points = self.arc_to_geodesic(self.inclines) \
                         * np.exp(1j * self.azimuths)
        return np.asarray(list(zip(
            np.real(complex_points),
            np.imag(complex_points),
            np.degrees(self.headings),
        )))

    # --- Lase Orderings ------------------------------------------------------
    def order_random_walk(self,
                          start_idx: int,
                          selection: Selection = None) -> List[int]:
        """An ordering of the geometry that starts at lase start_idx, and picks
        random neighbors to lase next until it runs out of neighbors, at which
        point is picks a random location to go to next. It does this until
        it has lased everything.

        Args:
            start_idx: The lase to start at.
            selection: A Selection to pick what you want to lase.
        """
        # Here's what we will lase
        lases = set(self.select(selection))
        if not lases:
            raise ValueError('Selecting an empty collection (nothing to lase).')

        # Logging information, in the l_ namespace
        l_log_every = min(1000, len(lases) // 5 + 1)  # Log every N steps
        l_num_complete = -1

        # Set up two lists which will always sum to everything we lase.
        ordering = []
        remaining = list(lases)

        # We shuffle the remaining so that when we run out of neighbors, we
        #  pick a random one.
        random.shuffle(remaining)

        # Manually do one iteration to start
        current_idx = start_idx
        remaining.remove(current_idx)
        ordering.append(current_idx)
        while remaining:
            # Logging information
            # We check if len(ordering) has decreased mod l_log_every. If so,
            #  that means over the last iteration, we passed a threshold.
            if len(ordering) % l_log_every < l_num_complete % l_log_every:
                l_percent_complete = 100 * len(ordering) / len(lases)
                _logger.info(f'{l_percent_complete:.2f}% complete...')
            l_num_complete = len(ordering)

            # Find neighbors that we haven't visited yet
            # It's cached, so we're not wasting time
            neighbors = set(self.neighbors_of(current_idx)) & lases
            neighbors = list(neighbors - set(ordering))

            if neighbors:
                # Pick a random remaining neighbor
                random.shuffle(neighbors)
                next_idx = neighbors[0]
            else:
                # Otherwise, pick a random spot
                next_idx = remaining[0]

            remaining.remove(next_idx)
            ordering.append(next_idx)
            current_idx = next_idx
        return ordering

    def order_spiral(self,
                     start_coords: XYZPoint = (0, 0, 1),
                     selection: Selection = None) -> List[int]:
        """An ordering of the geometry that starts at the lase closest to
        start_coords and spirals outwards. start_coords defaults to north.

        Args:
            start_coords: The coordinates to spiral from.
            selection: A Selection to pick what you want to lase.
        """
        lases = self.select(selection)
        lases_set = set(lases)
        if len(lases) == 0:
            raise ValueError('Selecting an empty collection (nothing to lase).')

        # Logging information, in the l_ namespace
        l_log_every = min(1000, len(lases) // 5 + 1)  # Log every N steps
        l_num_complete = -1

        # Find our start index
        dist_from_start = np.sum((self.lases - start_coords) ** 2, axis=1)
        dist_from_start[~self.mask(lases)] = float('inf')
        start_idx = np.argmin(dist_from_start)
        assert start_idx in lases, 'Start index not in selection.'

        # A function to find neighbors that are in the selection
        def neighbors_todo(lase_idx, distance):
            neighs = set(self.neighbors_of(lase_idx, distance=distance))
            neighs = list(neighs & lases_set)
            return neighs

        ordering = []
        level = 0
        # Run until we've ordered all the lases we want to.
        while lases_set - set(ordering):
            # Logging information
            # We check if len(ordering) has decreased mod l_log_every. If so,
            #  that means over the last iteration, we passed a threshold.
            if len(ordering) % l_log_every < l_num_complete % l_log_every:
                l_percent_complete = 100 * len(ordering) / len(lases)
                _logger.info(f'{l_percent_complete:.2f}% complete...')
            l_num_complete = len(ordering)

            # It's cached, so we're not wasting time
            neighbors = neighbors_todo(start_idx, distance=level)

            # Sort the neighbors by azimuth
            ordering.extend(
                sorted(neighbors, key=lambda idx: self.azimuths[idx])
            )
            level += 1
        return ordering

    # --- Toolpaths -----------------------------------------------------------
    @harness.toolpath
    def toolpath(
        self,
        order='spiral',
        fence_at=360,
        selection: Selection = None,
    ):
        """A toolpath for the LensGeometry.

        Args:
            order: The order in which to lase everything. Current options are
                'spiral' and 'random'.
            fence_at: The degrees at which, if N is moving more than it, it
                should insert fence instructions to break up the movement,
                hopefully keeping from bumping anything.
            selection: A Selection to pick what you want to lase.
        """
        # --- Digest/calculate the ordering -----------------------------------
        if isinstance(order, str):
            match order:
                case 'spiral':
                    ordering = self.order_spiral(selection=selection)
                case 'random':
                    ordering = self.order_random_walk(selection=selection)
                case _:
                    raise ValueError(f'Invalid ordering {order} supplied!')
        else:
            ordering = order

        # --- Calculate the coordinates to Lase at ----------------------------
        lase_coords = np.asarray(list(zip(
            self.inclines[ordering],
            self.azimuths[ordering],
            self.headings[ordering],
        )))
        lase_coords = np.degrees(lase_coords)

        # --- Toolpath --------------------------------------------------------
        yield In.FENCE
        incline_prev, azimuth_prev, heading_prev = (0, 0, 0)
        for lase_num, (incline, azimuth, heading) in enumerate(lase_coords):
            # --- Adjust azimuths to minimize movement ---
            azi_offsets = np.array(
                ((azimuth_prev // 360), (azimuth_prev // 360 + 1))) * 360
            azimuth = min(azi_offsets + azimuth,
                          key=lambda x: abs(x - azimuth_prev))

            # --- Interpolate large heading movements ---
            num_steps = int(abs(heading - heading_prev) // fence_at) + 1

            inclines = np.linspace(incline_prev, incline, num_steps,
                                   endpoint=False)
            azimuths = np.linspace(azimuth_prev, azimuth, num_steps,
                                   endpoint=False)
            headings = np.linspace(heading_prev, heading, num_steps,
                                   endpoint=False)

            for (incline_small, azimuth_small, heading_small) in zip(inclines,
                                                                     azimuths,
                                                                     headings):
                yield In.GO_SPH, azimuth_small, incline_small, heading_small
                yield In.FENCE

            # --- Move to final position & lase ---
            yield In.GO_SPH, azimuth, incline, heading
            yield In.LASE
            incline_prev, azimuth_prev, heading_prev = (
            incline, azimuth, heading)

    def toolpath_project_to_xyv(self, selection: Selection = None):
        """A toolpath which lases in XY the projection given by
        project_sph_to_xyv. This function is not meant for big lases, and
        doesn't allow you to pick the order you lase in."""
        ordering = self.order_spiral(selection=selection)
        return toolpaths.path_xyv_points(self.project_sph_to_xyv()[ordering])

    # --- Plotting ------------------------------------------------------------
    def plot(
            self,
            plot_lenslet: PlotArgs = True,
            plot_hexes: PlotArgs = True,
            plot_lases: PlotArgs = False,
            plot_zenith: PlotArgs = True,

            lase_labels=None,
            mark: Selection = None,

            jupyter_backend: str = 'panel'
    ):
        # Labels only work with trame
        if jupyter_backend != 'trame' and lase_labels is not None:
            _logger.warning(
                'Changing backend to trame, this is required for labeling '
                'lases. To silence this warning, do jupyter_backend=\'trame\'.')
            jupyter_backend = 'trame'

        # --- Handle inputs ---
        kwargs_lenslet = dict(style='surface', color='CFAF8F')
        kwargs_hexes = dict(style='surface', show_edges=True, color='444444')
        kwargs_lases = dict(style='points', color='black')
        kwargs_zenith = dict(style='points', color='green')

        if isinstance(plot_lenslet, dict): kwargs_hexes.update(kwargs_lenslet)
        if isinstance(plot_hexes, dict): kwargs_hexes.update(kwargs_hexes)
        if isinstance(plot_lases, dict): kwargs_hexes.update(kwargs_lases)
        if isinstance(plot_zenith, dict): kwargs_hexes.update(kwargs_zenith)

        kwargs_mark = copy.deepcopy(kwargs_hexes)
        kwargs_mark.update(dict(color='red'))
        do_mark = mark is not None and any(mark)

        # --- Create hex_vertices and hex_faces_pv, which plots the hexagons ---
        # Mark according to the mask
        if do_mark:
            mask = self.mask(mark)

            corners_reg = self.corners[~mask]
            corners_mark = self.corners[mask]
        else:
            corners_reg = self.corners

        hex_faces = np.arange(corners_reg.size // 3).reshape(corners_reg.shape[:-1])
        hex_vertices = corners_reg.reshape(-1, 3)
        hex_faces_pv = _pad_faces_for_pyvista(hex_faces)

        if do_mark:
            mark_faces = np.arange(corners_mark.size // 3).reshape(corners_mark.shape[:-1])
            mark_vertices = corners_mark.reshape(-1, 3)
            mark_faces_pv = _pad_faces_for_pyvista(mark_faces)

        # --- Create a plotter ---
        plotter = pv.Plotter(
            notebook=True,
            # window_size=(800, 800)
        )
        plotter.set_background('white')  # ('F0F0F0')

        # --- Create surfaces from stuff I already calculated ---
        hex_surface = pv.PolyData(hex_vertices, hex_faces_pv)
        if do_mark: mark_surface = pv.PolyData(mark_vertices, mark_faces_pv)
        lase_points = pv.PolyData(self.lases)
        zenith = pv.PolyData([(0.0, 0.0, 1.0)])
        lenslet = pv.Sphere(radius=0.99)

        # --- Add everything to the plotter
        if plot_lenslet: plotter.add_mesh(lenslet, **kwargs_lenslet)
        if plot_hexes: plotter.add_mesh(hex_surface, **kwargs_hexes)
        if plot_lases: plotter.add_mesh(lase_points, **kwargs_lases)
        if plot_zenith: plotter.add_mesh(zenith, **kwargs_zenith)
        if do_mark: plotter.add_mesh(mark_surface, **kwargs_mark)

        # --- Label the lase points ---
        if lase_labels is not None:
            lase_points['labels'] = lase_labels
            plotter.add_point_labels(
                lase_points, 'labels',
                # Point Options
                point_size=5, point_color='white',
                # Text Options
                font_size=15, text_color='black',
            )

        # --- Plot ---
        plotter.enable_trackball_style()
        plotter.show(jupyter_backend=jupyter_backend)

    def plot_edge_distances(self, goal: Optional[float] = None):
        """Plots self.get_edge_distances() in a histogram."""
        edge_dists = self.get_edge_distances()

        mean_dist = np.mean(edge_dists)
        median_dist = np.median(edge_dists)

        ax = plt.gca()
        ax.hist(edge_dists, bins=40)
        ax.set_ylabel('Count')
        ax.set_xlabel('Center-to-center distance (nm)')
        ax.axvline(mean_dist, color='black')
        ax.axvline(median_dist, color='red')
        ax.text(mean_dist + 0.5, ax.get_ylim()[1] - 100, 'mean', color='black',
                rotation='vertical')
        ax.text(median_dist + 0.5, ax.get_ylim()[1] - 100, 'median',
                color='red', rotation='vertical')
        if goal is not None:
            ax.axvline(goal, color='green')
            ax.text(goal + 0.5, ax.get_ylim()[1] - 100, 'goal', color='green',
                    rotation='vertical')

    def plot_min_lase_distances(self, goal: Optional[float] = None):
        """Plots self.get_min_lase_distances() in a histogram."""
        edge_dists = self.get_min_lase_distances()

        mean_dist = np.mean(edge_dists)
        median_dist = np.median(edge_dists)

        ax = plt.gca()
        ax.hist(edge_dists, bins=40)
        ax.set_ylabel('Count')
        ax.set_xlabel('Center-to-center distance (nm)')
        ax.axvline(mean_dist, color='black')
        ax.axvline(median_dist, color='red')
        ax.text(mean_dist + 0.5, ax.get_ylim()[1] - 100, 'mean', color='black',
                 rotation='vertical')
        ax.text(median_dist + 0.5, ax.get_ylim()[1] - 100, 'median',
                 color='red', rotation='vertical')
        if goal is not None:
            ax.axvline(goal, color='green')
            ax.text(goal + 0.5, ax.get_ylim()[1] - 100, 'goal', color='green',
                    rotation='vertical')

    # --- Magic Methods -------------------------------------------------------
    def __getitem__(self, index: Union[int, str]) -> XYZPoint:
        if isinstance(index, str):
            return self.lases[self.flags[index]]
        return self.lases[index]

    def __len__(self):
        return len(self.lases)

# --- Functions ----------------------------------------------------------------


def _pad_faces_for_pyvista(faces):
    padded_faces = []
    for face in faces:
        padded_face = [len(face)] + list(face)
        padded_faces.extend(padded_face)
    return padded_faces


def _make_geodesic_sphere(a_val, b_val, repeat=1, hemisphere=False):
    """Returns vertex locations, edge pairs (index, index), and face vertices (ind, ind, ind).
    I don't know what repeat does."""
    verts = []
    edges = {}
    faces = []
    geodesic.get_poly(
        poly='i',  # Icosahedron
        verts=verts,
        edges=edges,
        faces=faces
    )
    # Frequency = repeat * triangulation number (??)
    freq = repeat * (a_val ** 2 + a_val * b_val + b_val ** 2)
    # I think this is the subdivision of a triangle
    grid = geodesic.make_grid(freq, a_val, b_val)

    # I think this loop turns the icosahedron into the geodesic icosahedron we want
    for face in faces:
        verts.extend(geodesic.grid_to_points(
            grid=grid,
            freq=freq,
            div_by_len=False,
            f_verts=[verts[vert_index] for vert_index in face],
            face=face,
        ))

    # Project onto a sphere
    verts = [vert.unit() for vert in verts]

    # Convert to numpy
    verts = np.array([vert.v for vert in verts])

    # Remove half if desired
    if hemisphere:
        verts = [vert for vert in verts if vert[2] >= 0]

    # Create edges_of and faces using the convex hull
    chull = scipy.spatial.ConvexHull(points=verts)
    edges = set()
    faces = np.asarray(chull.simplices)

    # Faces are triples of indecies of vertexes
    #  So if a face is (A, B, C), then (A, B) is an edge
    for vindex1, vindex2, vindex3 in faces:
        edges.add(tuple(sorted((vindex1, vindex2))))
        edges.add(tuple(sorted((vindex2, vindex3))))
        edges.add(tuple(sorted((vindex1, vindex3))))

    return verts, np.asarray(list(edges)), np.asarray(chull.simplices)


def _make_corners_polygonal(num_sides, lases, headings, arc_radius):
    """Inscribes a n-gon of long radius arc_radius at each lase, where
    headings points towards one of the sides of each n-gon."""

    # --- N-gonal Geometry ---
    poly_angle = 2 * np.pi / num_sides

    # We are going to be finding the S corners of each of the N lases.
    # We're going to do this by drawing a circle around each lase,
    #  where the vertices of each polygon is in that circle.
    # We then take the top of that circle, called the "polygon top"
    #  and we rotate that around the lase by each azimuth that we want.
    nlases = len(lases)

    # --- Finding the azimuths; Shape (S, N) ---
    azimuths = np.vstack([headings] * num_sides)
    corner_angles = np.arange(0, num_sides) * poly_angle + poly_angle / 2
    for col, angle in enumerate(corner_angles):
        azimuths[col] += angle

    # --- Finding the polygon tops; Shape (N, 3) broadcast to (N, S, 3, 1) ---
    lase_x, lase_y, lase_z = lases[:,0], lases[:,1], lases[:,2]

    lase_inclines = np.arccos(lase_z)
    top_inclines = lase_inclines - arc_radius

    top_x = lase_x * np.sin(top_inclines) / np.sin(lase_inclines)
    top_y = lase_y * np.sin(top_inclines) / np.sin(lase_inclines)
    top_z = np.cos(top_inclines)

    polygon_tops = np.vstack([top_x, top_y, top_z]).T

    polygon_tops = np.broadcast_to(polygon_tops, (num_sides, nlases, 3))
    polygon_tops = np.swapaxes(polygon_tops, 0, 1)
    polygon_tops = np.reshape(polygon_tops, (nlases, num_sides, 3, 1))

    # --- Making 3x3 Identity matrices into Shape (N, num_sides, 3, 3) ---
    identity = np.eye(3)
    identity = np.broadcast_to(identity, (nlases, num_sides, 3, 3))

    # --- Make the cross matrices; Shape (N, 3, 3) broadcast to (N, S, 3, 3) ---
    cross_matrices = np.zeros(lases.shape + (3,))
    cross_matrices[:,0,1] = -lase_z
    cross_matrices[:,1,0] = lase_z
    cross_matrices[:,0,2] = lase_y
    cross_matrices[:,2,0] = -lase_y
    cross_matrices[:,1,2] = -lase_x
    cross_matrices[:,2,1] = lase_x

    cross_matrices = np.broadcast_to(cross_matrices, (num_sides, nlases, 3, 3))
    cross_matrices = np.swapaxes(cross_matrices, 0, 1)

    # --- Make the outer products; Shape (N, 3, 3) broadcast to (N, S, 3, 3) ---
    lases_cols = np.reshape(lases, (nlases, 3, 1))
    lases_rows = np.reshape(lases, (nlases, 1, 3))

    outer_products = lases_cols @ lases_rows

    outer_products = np.broadcast_to(outer_products, (num_sides, nlases, 3, 3))
    outer_products = np.swapaxes(outer_products, 0, 1)

    # --- Making cos & sin arrays; Shape (N) boradcast to (N, S, 3, 3) ---
    cosine = np.cos(azimuths)
    sine = np.sin(azimuths)

    cosine = np.broadcast_to(cosine, ((3, 3, num_sides, nlases)))
    cosine = np.moveaxis(cosine, [0, 1, 2, 3], [2, 3, 1, 0])
    sine = np.broadcast_to(sine, ((3, 3, num_sides, nlases)))
    sine = np.moveaxis(sine, [0, 1, 2, 3], [2, 3, 1, 0])

    # --- Finally making the rotation matrices; Shape (N, S, 3, 3) ---
    rotation_matrices = (cosine * identity) \
                        + (sine * cross_matrices) \
                        + ((1 - cosine) * outer_products)

    # --- Make the corners; Shape (N, S, 3) ---
    corners = np.reshape((rotation_matrices @ polygon_tops),
                         (nlases, num_sides, 3))

    return corners
