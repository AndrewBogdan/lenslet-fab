"""harness.py

Tools for more easily coordinating the laser and controller. Build around
generators of instructions; see llfab/toolpaths.py for examples.
"""

from typing import Generator, TypeAlias, Optional
from collections.abc import Iterator
import enum
import functools
import math
import re

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
import numpy as np
from numpy.typing import NDArray

from llfab import sal
from llfab.sal import Inst, Motors


Instruction: TypeAlias = tuple[Inst, ...]
SALPositionNP: TypeAlias = NDArray[np.float64]
SALPosition: TypeAlias = tuple[float, ...]


class _Toolpath(Iterator):
    """A custom iterator class to represent an iterator of tool instructions.
    In addition to acting like a normal iterator, has utilities for plotting,
    visualizing, error-checking, and toolpath simplification."""

    # Format strings for __repr__
    HEADER_FMT = '{0:^5} | {1:^5} | {2:^10} | {3:^10} |' \
                 ' {4:^10} | {5:^10} | {6:^10} | {7:^10}\n'
    ROW_FMT = '{0:^5} | {1:^5} | {2:>10.3f} | {3:>10.3f} |' \
              ' {4:>10.3f} | {5:>10.3f} | {6:>10.3f} | {7:>10.3f}\n'

    def __init__(self, inst_iterator: Iterator):
        """Make a toolpath, wrapping the given iterator. See `toolpath` for a
        description of how to generate the correct format.

        Args:
            inst_iterator: The iterator to wrap and modify.
        """
        insts, positions, lases = self._get_instructions(inst_iterator)
        self.insts = insts
        self.positions = positions
        self.lases = lases

        motor_positions = list(zip(Motors, zip(*self.positions)))
        self.required_motors = [mot for mot, positions in motor_positions
                                if any(positions)]

        self._pointer = 0
        self._next = None

        self._pos_num = 0
        self._lase_num = 0

        assert all(lase in self.positions for lase in self.lases), \
            'Toolpath.lases is not a subset of Toolpath.positions.'

    def __next__(self):
        """TODO"""
        try:
            self._next = self.insts[self._pointer]
            return self._next
        except IndexError:
            raise StopIteration()

    @staticmethod
    def _get_instructions(inst_iterator: Iterator):
        """Get the list of instructions corresponding to the toolpath."""
        insts: list[Instruction] = []
        positions: list[SALPosition] = [(0,) * 6]
        lases: list[SALPosition] = []

        pos: SALPositionNP = np.array((0.0,) * 6)

        def record_position():
            nonlocal pos
            # If we have changed positions, log the new position.
            if tuple(pos) != positions[-1]:
                positions.append(tuple(pos))
                insts.append((Inst.GO, *pos))

        for instruction in inst_iterator:
            match instruction:
                case (Inst.MOVE, *mov_by):
                    mov_by = np.array(mov_by)
                    mov_by.resize((6, ))
                    pos += mov_by

                case (Inst.GO, *pos_to):
                    pos_to = np.array(pos_to)
                    pos_to.resize((6, ))
                    pos = pos_to.copy()

                case (Inst.GO_SPH, *pos_sph_to):
                    pos_sph_to = np.array(pos_sph_to)
                    pos_sph_to.resize((3, ))
                    pos_to = np.array(sal.SALC.calc_spherical_pos(*pos_sph_to))
                    pos = pos_to.copy()

                case Inst.RETURN:
                    pos = np.array((0.0, ) * 6)
                    record_position()

                case Inst.FENCE:
                    # This doesn't actually correspond to any instruction,
                    #  it just separates movements (prevents them from being
                    #  rolled into one instruction).
                    record_position()

                case Inst.LASE:
                    record_position()
                    lases.append(tuple(pos))
                    insts.append(instruction)

                case _:
                    raise ValueError(
                        f'{instruction} is not a valid instruction.')

        return insts, positions, lases

    # --- Runtime Functionality -----------------------------------------------
    def confirm(self):
        """Confirm that the last instruction executed fully."""
        match self._next:
            case (Inst.MOVE | Inst.GO | Inst.GO_SPH, *_) | Inst.RETURN:
                self._pos_num += 1
            case Inst.LASE:
                self._lase_num += 1
        self._pointer += 1

    @property
    def lase_status(self):
        """Return a tuple of the number of lases completed and the total number
         of lases being done."""
        return self._lase_num, len(self.lases)

    @property
    def move_status(self):
        """Return a tuple of the number of moves completed and the total number
        of moves being done."""
        return self._pos_num, len(self.positions) - 1

    # --- Plotting ------------------------------------------------------------
    def plot(self, kind: Optional[str] = None, *args, **kwargs):
        """Plot the toolpath.

        Args:
            kind: The type of plot, chosen from 'xy', 'sph'. Defaults to 'xy'.
            *args: Arguments, passed to the plotting function.
            **kwargs: Keyword arguments, passed to the plotting function

        Returns:
            The axis plotted on.
        """
        kind = kind or 'xy'
        match kind:
            case 'xy':
                return self.plot_xy(*args, **kwargs)
            case 'sph':
                return self.plot_sph(*args, **kwargs)

    def plot_xy(self, ax=None, *,
                lase_colors=(
                    # '#9BC53D', '#508484', '#7E52A0', '#4A4238', '#E63946',
                    'green', 'yellowgreen', 'yellow', 'orange', 'orangered',
                ),
                args_path=None,
                args_lase=None,
                args_start=None,
                args_end=None):
        """Plot the toolpath in the Cartesian plane with X and Y.

        Args:
            ax: The matplotlib axis to plot on. Will try to a square one by
                default.
            lase_colors: (a sequence of 5 matplotlib colors) The colors of the
                lase markers, in the order (starting lase, finished lases,
                current lase, future lases, last lase).
            args_path: Arguments for the path line, passed to plt.plot.
            args_lase: Arguments for the lase markers, passed to plt.scatter.
            args_start: Arguments for the start position marker, passed to
                plt.scatter.
            args_end: Arguments for the end position marker, passed to
                plt.scatter.

        Returns:
            The axis ax, whichever was plotted on.
        """
        args_path = args_path or {}
        args_lase = args_lase or {}
        args_start = args_start or {}
        args_end = args_end or {}

        if ax is None:
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')

        # These represent the current state, and if we're finished, there's
        #  special behavior.
        pos_num = self._pos_num if self._pos_num != len(self.positions) else -1
        lase_num = self._lase_num if self._lase_num != len(self.lases) else -1

        pos_x, pos_y, _, _, _, _ = zip(*self.positions[pos_num:])
        lase_x, lase_y, _, _, _, _ = zip(*self.lases)

        # Change from um to m
        pos_x = np.array(pos_x) / 1e6
        pos_y = np.array(pos_y) / 1e6
        lase_x = np.array(lase_x) / 1e6
        lase_y = np.array(lase_y) / 1e6

        # Color-code the lases
        lc_first, lc_done, lc_current, lc_todo, lc_last = lase_colors
        # TODO: Should we switch to lase_num here?
        lase_c = [lc_done] * self._lase_num + \
                 [lc_todo] * (len(self.lases) - self._lase_num)
        lase_c[lase_num] = lc_current
        lase_c[0] = lc_first
        lase_c[-1] = lc_last

        # Plot the positions, the actual tool's path, as a line
        ax.plot(
            pos_y, pos_x,
            color=args_path.pop('color', '#0C0A3E'),
            **args_path
        )

        # Plot the lases
        ax.scatter(
            lase_y, lase_x,
            s=args_lase.pop('s', 200),
            c=args_lase.pop('c', lase_c),
            marker=args_lase.pop('marker', 'H'),
            **args_lase
        )

        # Plot the start and end points
        ax.scatter(
            [self.positions[-1][1]], [self.positions[-1][0]],
            s=args_end.pop('s', 50),
            c=args_end.pop('c', 'red'),
            marker=args_end.pop('marker', 's'),
            **args_end
        )
        ax.scatter(
            [self.positions[0][1]], [self.positions[0][0]],
            s=args_start.pop('s', 100),
            c=args_start.pop('c', 'green'),
            marker=args_start.pop('marker', '>'),
            **args_start
        )

        ax.xaxis.set_major_formatter(ticker.EngFormatter(unit='m'))
        ax.yaxis.set_major_formatter(ticker.EngFormatter(unit='m'))

        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.invert_xaxis()

        return ax

    def plot_sph(self, ax=None, *,
                 radius=1,
                 lase_num=0,  # TODO: Temporary

                 # Plotting parameters
                 lase_colors=(
                     # '#9BC53D', '#508484', '#7E52A0', '#4A4238', '#E63946',
                     'green', 'yellowgreen', 'yellow', 'orange', 'orangered',
                 ),
                 args_lase=None,
                 args_surface=None,
                 ):
        """TODO"""
        # --- Manage optional arguments ---
        args_lase = args_lase or {}
        args_surface = args_surface or {}

        # --- Make a plot of a hemisphere with radius ---
        if ax is None:
            _, ax = plt.subplots(
                subplot_kw={"projection": "3d"},
                figsize=(10, 7)
            )

        radius_shrunk = radius * 0.97  # Shrink the radius so we can see
        azimuth, incline = np.mgrid[0:2 * np.pi:20j, 0:np.pi / 2:20j]
        mesh_x = np.cos(azimuth) * np.sin(incline) * radius_shrunk
        mesh_y = np.sin(azimuth) * np.sin(incline) * radius_shrunk
        mesh_z = np.cos(incline) * radius_shrunk

        # --- Get the lase locations on the surface of the hemisphere ---
        lase_x = []
        lase_y = []
        lase_z = []
        for lase in self.lases:
            x_um, y_um, z_um, n_deg, p_deg, v_deg = lase

            incline = math.radians(p_deg)
            azimuth = math.radians(v_deg)

            orig_x, orig_y, orig_z = sal.SALC.calc_origin(*lase)

            # We actually lase above the origin, by radius
            lase_x.append(
                orig_x + math.cos(azimuth) * math.sin(incline) * radius)
            lase_y.append(
                orig_y + math.sin(azimuth) * math.sin(incline) * radius)
            lase_z.append(orig_z + math.cos(incline) * radius)

        # --- Change from um to m ---
        mesh_x = np.array(mesh_x) / 1e6
        mesh_y = np.array(mesh_y) / 1e6
        mesh_z = np.array(mesh_z) / 1e6
        lase_x = np.array(lase_x) / 1e6
        lase_y = np.array(lase_y) / 1e6
        lase_z = np.array(lase_z) / 1e6

        # --- Color-code the lases ---
        lase_num = lase_num if lase_num != len(self.lases) else -1
        lc_first, lc_done, lc_current, lc_todo, lc_last = lase_colors
        lase_c = [lc_done] * lase_num + \
                 [lc_todo] * (len(self.lases) - lase_num)
        lase_c[lase_num] = lc_current
        lase_c[0] = lc_first
        lase_c[-1] = lc_last

        # --- Plot everything ---
        ax.plot_surface(
            mesh_x, mesh_y, mesh_z,
            cmap=args_surface.pop('cmap', cm.summer),
            antialiased=args_surface.pop('antialiased', False),
            **args_surface
        )
        ax.scatter(
            lase_x, lase_y, lase_z,
            c=args_lase.get('c', lase_c),
            **args_lase,
        )

        # --- Set units ---
        ax.xaxis.set_major_formatter(ticker.EngFormatter(unit='m'))
        ax.yaxis.set_major_formatter(ticker.EngFormatter(unit='m'))
        ax.zaxis.set_major_formatter(ticker.EngFormatter(unit='m'))

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        return ax

    def __repr__(self):
        """Prints the positions and lases in a text table."""
        # Print a header for the table
        header = self.HEADER_FMT.format(
            'Pos#', 'Lase#',
            f'X ({sal.SALC.X_UNIT[1]})',
            f'Y ({sal.SALC.Y_UNIT[1]})',
            f'Z ({sal.SALC.Z_UNIT[1]})',
            f'N ({sal.SALC.N_UNIT[1]})',
            f'P ({sal.SALC.P_UNIT[1]})',
            f'V ({sal.SALC.V_UNIT[1]})',
        )
        string = header

        # Divider row
        string += re.sub(
            pattern=r'[^|\n]',
            repl='–',
            string=header,
        )

        # Print all the values
        for pos_num, position in enumerate(self.positions):
            x, y, z, n, p, v = position

            try:
                lase_num = self.lases.index(position)
            except ValueError:
                lase_num = '–––'

            string += self.ROW_FMT.format(pos_num, lase_num, x, y, z, n, p, v)
        return string

    __str__ = __repr__


def toolpath(inst_gen_function: Generator):
    """A decorator to apply to a generator of instructions. The decorated
    generator will generate a special toolpath iterator, which has utilities
    for plotting, visualizing, error-checking, and will be automatically
    simplified.

    The generator should yield instructions of the formats:
    - (Inst.MOVE, X (float, required), Y (optional), Z, N, P, V)
    - (Inst.GO, X (float, required), Y (optional), Z, N, P, V)
    - (Inst.GO_SPH, azimuth, incline, height)
        Both azimuth and incline are in degrees and are required, height is
        optional.
    - Inst.LASE
    - Inst.FENCE
    - Inst.RETURN

    Args:
        inst_gen_function: The generator function. See llfab/toolpaths.py for
            examples.

    Returns: A decorated version of the given generator which, when called,
        will return an _Toolpath object implementing the Iterator protocol.
    """

    @functools.wraps(inst_gen_function)
    def decorated_path(*args, **kwargs):
        return _Toolpath(inst_gen_function(*args, **kwargs))

    return decorated_path
