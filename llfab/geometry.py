"""Tools for geometric manipulation and creation of the lenslet geometry."""

# --- Libraries ---
from typing import TypeAlias, Union
import collections
import copy
import time
import functools
import logging
import math
import numbers

from typing import TypeVar
from collections.abc import Collection, Sequence
from numbers import Real

import numpy as np
import pyvista as pv
import scipy

from llfab import geodesic


# Type Aliases
SphPoint = TypeVar('SphPoint')
PlotArgs: TypeAlias = Union[bool, dict]

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
        if 'all' in kind or 'graph' in kind:
            self.edges_of.cache_clear()
            self.neighborhood_of.cache_clear()
            self.neighbors_of.cache_clear()
            self.degree_of.cache_clear()
            self.n_gons.cache_clear()
            self.get_edge_indices_map.cache_clear()

    # --- Utility -------------------------------------------------------------
    #  Graph theoretic tools
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

    @functools.lru_cache(maxsize=None)
    def sph(self, index):
        return SphPoint.from_xyz(*self.lases[index])

    # --- Stateless Geometric Calculations / Optimization Routines ------------

    # --- Stateful Geometric Calculations -------------------------------------
    def bound_headings(self, bounds):
        """Makes sure that abs(headings) is within bounds, and changes it
        if necessary."""
        mask = abs(self.headings) > bounds
        self.headings[mask] = bounds[mask] * np.sign(self.headings[mask])
        return self.headings

    def make_corners_hexagonal(self, hex_diameter_long: float):
        """Make the corners, assuming regular polygonal lases."""
        time_start = time.time()
        _logger.debug(f'Making hexagonal corners...')
        if self.lases is None: raise ValueError('Lases undefined!')
        if self.headings is None: raise ValueError('Headings undefined!')

        arc_radius = np.arcsin(hex_diameter_long / self.lens_diameter)
        self.corners = _make_corners_hexagonal(self.lases, self.headings, arc_radius)

        _logger.debug(f'Done! Time elapsed: '
                      f'{(time.time() - time_start) * 1e3:.2f} ms')
        return self.corners

    def make_lases_geodesic(self, a_val, b_val):
        """Make a LaseGeomtry where the lases are at the vertices of a geodesic sphere."""
        _logger.info(f'Making Geodesic Sphere GS({a_val}, {b_val}).')
        lases, edges, _ = _make_geodesic_sphere(a_val, b_val, repeat=1)
        self.lases = lases
        self.edges = edges
        return self.lases

    def make_headings_from_edges(self):
        """Picks headings so that they point towards an edge."""
        _logger.warning(f'Using inefficient method '
                        f'{self.__class__.__name__}.make_headings_from_edges:')
        if self.lases is None: raise ValueError('Lases undefined!')

        headings = []
        for lase_index, lase_xyz in enumerate(self.lases):
            if lase_index % min(1000, len(self.lases) // 5) == 0:
                _logger.info(f'\t{100 * lase_index / len(self.lases):.2f}% complete...')

            lase = self.sph(lase_index)
            lase_headings = []

            # Get the headings from each neighbor
            for neighbor_index in self.neighbors_of(lase_index):
                neighbor = self.sph(neighbor_index)
                lase_headings.append(lase.heading_to(neighbor))

            # Pick the one that's closest to North
            min_heading = min(lase_headings, key=abs)
            # if abs(min_heading) > 2 * np.pi / 12:
            #     _logger.warning(f'Lase #{lase_index} with min '
            #                     f'{np.degrees(min_heading):.2f} '
            #                     f'has possible headings '
            #                     f'{np.degrees(lase_headings)}, none are within range.')
            headings.append(min_heading)
        self.headings = np.array(headings)
        return self.headings

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
        return self.headings

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
        self.clear_caches('all')
        _logger.info('Saving pentagons and hexagons.')
        self.pentagons = set()
        self.hexagons = set(range(len(self.lases)))

    # --- Plotting ------------------------------------------------------------
    def plot(
            self,
            plot_lenslet: PlotArgs = True,
            plot_hexes: PlotArgs = True,
            plot_lases: PlotArgs = False,
            plot_zenith: PlotArgs = True,

            lase_labels=None,
            mark_mask=None,

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
        kwargs_zenith = dict(style='points', color='red')

        if isinstance(plot_lenslet, dict): kwargs_hexes.update(kwargs_lenslet)
        if isinstance(plot_hexes, dict): kwargs_hexes.update(kwargs_hexes)
        if isinstance(plot_lases, dict): kwargs_hexes.update(kwargs_lases)
        if isinstance(plot_zenith, dict): kwargs_hexes.update(kwargs_zenith)

        kwargs_mark = copy.deepcopy(kwargs_hexes)
        kwargs_mark.update(dict(color='red'))
        do_mark = mark_mask is not None and any(mark_mask)

        # --- Create hex_vertices and hex_faces_pv, which plots the hexagons ---
        # Mark according to the mask
        if do_mark:
            corners_reg = self.corners[~mark_mask]
            corners_mark = self.corners[mark_mask]
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

    # --- Magic Methods -------------------------------------------------------
    def __getitem__(self, index) -> SphPoint:
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


def _make_corners_hexagonal(lases, headings, arc_radius):
    """Incribes a hexagon of long radius arc_radius at each lase, where
    headings points towards one of the sides of each hexagon."""

    # --- Hexagonal Geometry ---
    POLY_ANGLE = 2 * np.pi / 6
    NUM_SIDES = 6

    # We are going to be finding the 6 corners of each of the N lases.
    # We're going to do this by drawing a circle around each lase,
    #  where the vertices of each hexagon is in that circle.
    # We then take the top of that circle, called the "hexagon top"
    #  and we rotate that around the lase by each azimuth that we want.
    nlases = len(lases)

    # --- Finding the azimuths; Shape (6, N) ---
    azimuths = np.vstack([headings] * 6)
    corner_angles = np.arange(0, NUM_SIDES) * POLY_ANGLE + POLY_ANGLE / 2
    for col, angle in enumerate(corner_angles):
        azimuths[col] += angle

    # --- Finding the hexagon tops; Shape (N, 3) broadcast to (N, 6, 3, 1) ---
    lase_x, lase_y, lase_z = lases[:,0], lases[:,1], lases[:,2]

    lase_inclines = np.arccos(lase_z)
    top_inclines = lase_inclines - arc_radius

    top_x = lase_x * np.sin(top_inclines) / np.sin(lase_inclines)
    top_y = lase_y * np.sin(top_inclines) / np.sin(lase_inclines)
    top_z = np.cos(top_inclines)

    hexagon_tops = np.vstack([top_x, top_y, top_z]).T

    hexagon_tops = np.broadcast_to(hexagon_tops, (6, nlases, 3))
    hexagon_tops = np.swapaxes(hexagon_tops, 0, 1)
    hexagon_tops = np.reshape(hexagon_tops, (nlases, 6, 3, 1))

    # --- Making 3x3 Identity matrices into Shape (N, 6, 3, 3) ---
    identity = np.eye(3)
    identity = np.broadcast_to(identity, (nlases, 6, 3, 3))

    # --- Make the cross matrices; Shape (N, 3, 3) broadcast to (N, 6, 3, 3) ---
    cross_matrices = np.zeros(lases.shape + (3,))
    cross_matrices[:,0,1] = -lase_z
    cross_matrices[:,1,0] = lase_z
    cross_matrices[:,0,2] = lase_y
    cross_matrices[:,2,0] = -lase_y
    cross_matrices[:,1,2] = -lase_x
    cross_matrices[:,2,1] = lase_x

    cross_matrices = np.broadcast_to(cross_matrices, (6, nlases, 3, 3))
    cross_matrices = np.swapaxes(cross_matrices, 0, 1)

    # --- Make the outer products; Shape (N, 3, 3) broadcast to (N, 6, 3, 3) ---
    lases_cols = np.reshape(lases, (nlases, 3, 1))
    lases_rows = np.reshape(lases, (nlases, 1, 3))

    outer_products = lases_cols @ lases_rows

    outer_products = np.broadcast_to(outer_products, (6, nlases, 3, 3))
    outer_products = np.swapaxes(outer_products, 0, 1)

    # --- Making cos & sin arrays; Shape (N) boradcast to (N, 6, 3, 3) ---
    cosine = np.cos(azimuths)
    sine = np.sin(azimuths)

    cosine = np.broadcast_to(cosine, ((3, 3, 6, nlases)))
    cosine = np.moveaxis(cosine, [0, 1, 2, 3], [2, 3, 1, 0])
    sine = np.broadcast_to(sine, ((3, 3, 6, nlases)))
    sine = np.moveaxis(sine, [0, 1, 2, 3], [2, 3, 1, 0])

    # --- Finally making the rotation matrices; Shape (N, 6, 3, 3) ---
    rotation_matrices = (cosine * identity) + (sine * cross_matrices) + ((1 - cosine) * outer_products)

    # --- Make the corners; Shape (N, 6, 3) ---
    corners = np.reshape((rotation_matrices @ hexagon_tops), (nlases, 6, 3))

    return corners
