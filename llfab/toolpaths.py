"""toolpaths.py

A collection of toolpaths to use. TODO: How do you use them?
"""

import math

import numpy as np

from llfab import harness
from llfab.harness import Inst as In


@harness.toolpath
def path_xy_grid(
    shape: tuple[int, int] = (10, 10),
    stride: tuple[float, float] = (1000.0, 1000.0),
):
    """Lase in a rectangular grid.

    Args:
        shape: (pair of ints) the number of rows and columns to lase in,
            formatted as (rows, columns).
        stride: (pair of floats) The distance between lases, in both x and y,
            formatted as (stride_x, stride_y)
    """
    rows, cols = shape
    stride_x, stride_y = stride

    for _ in range(rows):
        for _ in range(cols):
            yield In.LASE
            yield In.MOVE, 0, stride_y
        yield In.MOVE, stride_x, 0
        yield In.MOVE, 0, -1 * stride_y * cols
    yield In.RETURN


@harness.toolpath
def path_xy_hex_grid(
    dimensions: tuple[float, float] = (1000.0, 1000.0),
    pitch: tuple[float, float] = (100.0, 115.47),
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
            yield In.LASE
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
    pitch: tuple = (100.0, 115.47),
):
    """Lase a circular region in a hexagonal pattern.

    Args:
        radius: (float) The radius in microns around the starting point to fill.
        pitch: (pair of floats) The dimensions of the hexagonal pattern, in
            microns. See `path_xy_hex_grid` for a guide on what to supply.
    """
    radius = float(radius)
    pitch_x = float(pitch[0])
    pitch_y = float(pitch[1])

    if radius < 0:
        raise ValueError('Radius cannot be negative.')
    if pitch_x < 0 or pitch_y < 0:
        raise ValueError('Pitches cannot be negative.')

    # Move to upper right corner of a square
    pos = np.array((radius, radius))
    yield In.MOVE, radius, radius

    direction = -1
    while pos[0] >= -radius:
        while pos[1] * direction <= radius:
            # Lase only if we're inside the circle
            if math.sqrt(pos[0] ** 2 + pos[1] ** 2) <= radius:
                yield In.LASE
            # Move to the left or right in Y
            mov = np.array((0, pitch_y * direction))
            pos += mov
            yield In.MOVE, *mov
        # Move down in X, back half of Y
        mov = np.array((-pitch_x, pitch_y / 2 * direction * -1))
        pos += mov
        yield In.MOVE, *mov
        # Change direction
        direction *= -1
    yield In.RETURN


@harness.toolpath
def path_xy_circle(
    radius: float = 500.0,
    stride: float = 100.0,
):
    """Lase in a circle, centered at the starting point.

    Args:
        radius: (float) The radius of the circle, in microns.
        stride: (float) The distance (along the circle) between lases. For
            example, a unit circle with stride of pi would have two lases in
            total, one at 0 and one at 180 degrees.
    """

    stride_angle = stride / radius
    angle = 0

    while angle < 2 * math.pi:
        yield In.GO, math.cos(angle) * radius, math.sin(angle) * radius
        angle += stride_angle
        yield In.LASE
    yield In.RETURN
