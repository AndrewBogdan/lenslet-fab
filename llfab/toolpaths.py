"""toolpaths.py

A collection of toolpaths to use. TODO: How do you use them?
"""

import math

import numpy as np

from llfab import harness
from llfab import util
from llfab.harness import Inst as In


def _lase_z_steps(z_steps: tuple):
    """Lase several times at the same XY-location, changing z-height according
    to z_steps.

    Args:
        z_steps: A tuple of floats of how to move the Z-Axis between lases. If
            empty, it will lase once without moving.

    """
    yield In.LASE

    for z_step in z_steps:
        yield In.MOVE, 0, 0, z_step
        yield In.LASE
    yield In.MOVE, 0, 0, -sum(z_steps)


def _get_hex_grid(
    pitch: float,
    radius: float,
    angle: float,
) -> np.ndarray:
    """Get a hexagonal grid of the given radius, rotated by the given angle.

    Args:
        pitch: The distance between two adjacent hexagons.
        radius: The points will all have magnitude less than this radius, and
            The area within this radius will be filled with points.
        angle: An angle of zero will correspond to hexagons oriented like ⬢.
            That is, along the x-axis (y=0), it will appear as ⬢⬢⬢⬢⬢.

    Returns: A numpy array of points, sorted by x first, and then by y. One of
        the points will be (0, 0), within 1e-6 error.
    """
    if pitch < 0: raise ValueError('Pitch cannot be negative.')
    if radius < 0: raise ValueError('Radius cannot be negative.')

    pitch_short = pitch * math.sqrt(3) / 2
    pitch_long = pitch
    grid_diameter = 2 * radius + 3 * pitch_long

    num_short = math.floor(grid_diameter / pitch_short)
    num_long = math.floor(grid_diameter / pitch_long)

    # Introduce short and long axes
    pos_short = -grid_diameter / 2
    pos_long = -grid_diameter / 2

    # Create a hexagonal grid
    positions = []
    direction = 1
    for _ in range(num_short):
        for _ in range(num_long):
            # Record a position
            positions.append((pos_long, pos_short))
            pos_long += pitch_long * direction
        # Move down in Short, back half of Long
        pos_short += pitch_short
        pos_long += pitch_long / 2 * direction * -1
        # Change direction
        direction *= -1
    positions = np.array(positions)

    # Align the center of the grid to (0, 0)
    center_index = np.argmin(np.linalg.norm(positions, axis=1))
    positions = positions - positions[center_index]

    # Rotate the grid by angle
    positions_complex = (positions[:, 0] + 1j * positions[:, 1]) * np.exp(1j * np.radians(angle))
    positions = np.array([np.real(positions_complex), np.imag(positions_complex)]).T

    # Remove everything outside of the radius
    positions = positions[np.linalg.norm(positions, axis=1) < radius]
    assert any(np.linalg.norm(positions, axis=1) < 1e-6)

    # Sort
    positions = np.array(sorted(positions.tolist()))
    return positions


@harness.toolpath
def path_xyv_points(points: np.ndarray):
    """Make a toolpath out of an array of points.

    Args:
        points: np.ndarray of shape (N, 3), such that
        - points[:, 0] are the X-coordinates,
        - points[:, 1] are the Y-coordinates, and
        - points[:, 2] are the headings, i.e. V-coordiantes.
    """
    for x, y, v in points:
        yield In.GO, x, y, None, None, None, v
        yield In.LASE
    yield In.RETURN


@harness.toolpath
def path_xy_grid(
    shape: tuple[int, int] = (10, 10),
    stride: tuple[float, float] = (1000.0, 1000.0),
    z_steps: tuple = (),
):
    """Lase in a rectangular grid.

    Args:
        shape: (pair of ints) the number of rows and columns to lase in,
            formatted as (rows, columns).
        stride: (pair of floats) The distance between lases, in both x and y,
            formatted as (stride_x, stride_y)
        z_steps: A tuple of floats of how to move the Z-Axis between lases. If
            empty (the default), it will lase once without moving.
    """
    rows, cols = shape
    stride_x, stride_y = stride

    for _ in range(rows):
        for _ in range(cols):
            yield from _lase_z_steps(z_steps)
            yield In.MOVE, 0, stride_y
        yield In.MOVE, stride_x, 0
        yield In.MOVE, 0, -1 * stride_y * cols
    yield In.RETURN


@harness.toolpath
@util.depreciate
def path_xy_hex_grid(
    dimensions: tuple[float, float] = (1000.0, 1000.0),
    pitch: tuple[float, float] = (100.0, 115.47),
    z_steps: tuple = (),
):
    """Lase a rectangular region in a hexagonal pattern.

    Args:
        dimensions: (pair of floats) The x and y size of the rectangle to fill,
            in microns. For example, the default is (1000.0, 1000.0), which
            means that, the lasing will be contained within a 1cm by 1cm region.

        pitch: (pair of floats) The dimensions of the hexagonal pattern, in
            microns.

              ⬡   ⬡   ⬡ |
            ⬡   ⬡   ⬡   | These lines total one pitch in Y
              ⬡   ⬡   ⬡ |
            ⬡   ⬡   ⬡
            |----|
            This is one pitch in X
            
            Notice the orientation of the hexagon symbols. If you were trying
            to fill space with hexagons that were 1cm (1000 microns) on their
            shorter (inscribed circle) diameter, the pitch in X would be 1000
            microns, and the pitch in Y would be 1154.7 microns, approximately
            pitch_y = (2 / sqrt(3)) * pitch_x.

            Your input should be in the format (pitch_x, pitch_y).

        z_steps: A tuple of floats of how to move the Z-Axis between lases. If
            empty (the default), it will lase once without moving.
    """
    dim_x, dim_y = dimensions
    pitch_x, pitch_y = pitch

    num_x = math.floor(dim_x / pitch_x)
    num_y = math.floor(dim_y / pitch_y)

    if dim_x < 0 or dim_y < 0:
        raise ValueError('Dimensions cannot be negative.')
    if pitch_x < 0 or pitch_y < 0:
        raise ValueError('Pitches cannot be negative.')

    # Move to the upper right corner, so the grid is centered on zero.
    yield In.MOVE, (dim_x - pitch_x)/2, (dim_y - pitch_y)/2

    direction = -1
    for _ in range(num_x):
        for _ in range(num_y):
            yield from _lase_z_steps(z_steps)
            # Move to the left or right in Y
            yield In.MOVE, 0, pitch_y * direction
        # Move down in X, back half of Y
        yield In.MOVE, -pitch_x, pitch_y / 2 * direction * -1
        # Change direction
        direction *= -1
    yield In.RETURN


@harness.toolpath
def path_xy_hex_grid_circle(
    radius: float = 500.0,
    pitch: float = 100.0,
    angle: float = 0.0,
    z_steps: tuple = (),
):
    """Lase a circular region in a hexagonal pattern.

    Args:
        radius: (float) The radius in microns around the starting point to fill.
        pitch: (pair of floats) The dimensions of the hexagonal pattern, in
            microns. See `path_xy_hex_grid` for a guide on what to supply.
        z_steps: A tuple of floats of how to move the Z-Axis between lases. If
            empty (the default), it will lase once without moving.
    """
    radius = float(radius)
    pitch = float(pitch)
    angle = float(angle)

    lases = _get_hex_grid(pitch=pitch, radius=radius, angle=angle)

    for (lase_x, lase_y) in lases:
        yield In.GO, lase_x, lase_y
        yield from _lase_z_steps(z_steps)
    yield In.RETURN


@harness.toolpath
def path_xy_circle(
    radius: float = 500.0,
    stride: float = 100.0,
    z_steps: tuple = (),
):
    """Lase in a circle, centered at the starting point.

    Args:
        radius: (float) The radius of the circle, in microns.
        stride: (float) The distance (along the circle) between lases. For
            example, a unit circle with stride of pi would have two lases in
            total, one at 0 and one at 180 degrees.
        z_steps: A tuple of floats of how to move the Z-Axis between lases. If
            empty (the default), it will lase once without moving.
    """

    stride_angle = stride / radius
    angle = 0

    while angle < 2 * math.pi:
        yield In.GO, math.cos(angle) * radius, math.sin(angle) * radius
        angle += stride_angle
        yield from _lase_z_steps(z_steps)
    yield In.RETURN


@harness.toolpath
def path_sph_hex_grid(
        radius: float = 1000.0,
        pitch: tuple = (100.0, 115.47),
        max_incline: float = 90.0,
):
    # --- Unpack inputs and validate ---
    pitch_x, pitch_y = pitch

    if radius < 0:
        raise ValueError('Radius cannot be negative.')
    if pitch_x < 0 or pitch_y < 0:
        raise ValueError('Pitches cannot be negative.')

    # --- Create grid of hexagons for polar projection ---
    rel_pitch_x = (pitch_x / radius)
    rel_pitch_y = (pitch_y / radius)
    num_x = math.floor(4 / rel_pitch_x)
    num_y = math.floor(4 / rel_pitch_y)

    # Create a grid of hexagons one at a time
    # TODO: this code could probably use some shortening
    positions = []
    pos_x, pos_y = (4 - rel_pitch_x) / 2, (4 - rel_pitch_y) / 2

    direction = -1
    for _ in range(num_x):
        for _ in range(num_y):
            positions.append((pos_x, pos_y))
            # Move to the left or right in Y
            pos_y += rel_pitch_y * direction
        # Move down in X, back half of Y
        pos_x += -rel_pitch_x
        pos_y += rel_pitch_y / 2 * direction * -1
        # Change direction
        direction *= -1
    x, y = zip(*positions)

    # Adjust the grid so (0, 0) has a point
    x = np.asarray(x)
    y = np.asarray(y)
    mindex = np.argmin(x ** 2 + y ** 2)
    x = x - x[mindex]
    y = y - y[mindex]

    # --- Sort the points radially ---
    pts = list(zip(x, y, x ** 2 + y ** 2, np.angle(x + 1j * y)))
    pts = sorted(pts, key=lambda x: x[2])
    sort = []
    sort.append(pts.pop(0))
    while len(pts) != 0:
        ref_x, ref_y, ref_m, ref_a = sort[-1]

        def hex_dist(p):
            return (p[0] - ref_x) ** 2 + (p[1] - ref_y) ** 2

        dists = [hex_dist(p) for p in pts]
        mind = min(dists)
        closest_mask = [abs(d - mind) < 1e-6 for d in dists]
        closest = [p for ind, p in enumerate(pts) if closest_mask[ind]]
        nex = min(closest, key=lambda p: p[2])
        sort.append(nex)
        pts.remove(nex)
    x, y, _, _ = list(zip(*sort))

    @np.vectorize
    def norm_proj(x, y):
        return np.sqrt(x ** 2 + y ** 2), np.angle(x + 1j * y)

    incs, azis = norm_proj(x, y)
    incs = np.degrees(incs)
    azis = np.degrees(azis)

    # --- Make azimuths increasing (to prevent going from 179 to -179) ---
    prev_azi = azis[0]
    clean_azis = [prev_azi]
    for azi in azis[1:]:
        rotations, offset = divmod(prev_azi, 360)
        residue = np.asarray([azi - 360, azi, azi + 360]) + rotations * 360
        mindex = np.argmin(np.abs(prev_azi - residue))
        clean_azis.append(residue[mindex])
        prev_azi = clean_azis[-1]
    azis = clean_azis

    # --- Create instructions ---
    points = [(inc, azi) for inc, azi in zip(incs, azis) if inc <= max_incline]

    for incline, azimuth, in points:
        yield In.GO_SPH, azimuth, -incline
        yield In.LASE

    yield In.GO, None, None, 0, None, None, None
    yield In.FENCE
    yield In.RETURN