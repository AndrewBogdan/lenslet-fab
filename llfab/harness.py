"""harness.py

Tools for more easily coordinating the laser and controller. Build around
generators of instructions; see llfab/toolpaths.py for examples.
"""

from typing import Optional, Generator, TypeAlias, ParamSpec
from collections.abc import Iterator
import enum
import functools
import math

from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from llfab import motor


class Inst(enum.Enum):
    MOVE = 'MOVE'
    GO = 'GO'
    LASE = 'LASE'

    # Psuedo-instructions/Directives
    FENCE = 'FENCE'
    RETURN = 'RETURN'


Instruction: TypeAlias = tuple[Inst, ...]
SALPositionNP: TypeAlias = NDArray[np.float64]
SALPosition: TypeAlias = tuple[float, ...]


class _Toolpath(Iterator):
    """A custom iterator class to represent an iterator of tool instructions.
    In addition to acting like a normal iterator, has utilities for plotting,
    visualizing, error-checking, and toolpath simplification."""

    def __init__(self, inst_iterator: Iterator):
        """Make a toolpath, wrapping the given iterator. See `toolpath` for a
        description of how to generate the correct format.

        Args:
            inst_iterator: The iterator to wrap and modify.
        """
        insts, positions, lases = self._get_instructions(inst_iterator)
        self._insts = insts
        self._positions = positions
        self._lases = lases

        self._pointer = 0
        self._next = None

        self._pos_num = 0
        self._lase_num = 0

    def __next__(self):
        """TODO"""
        try:
            self._next = self._insts[self._pointer]
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

                case Inst.RETURN:
                    pos = np.array((0.0, ) * 6)
                    record_position()
                    insts.append((Inst.GO, *pos.copy()))

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

    def confirm(self):
        """Confirm that the last instruction executed fully."""
        match self._next:
            case (Inst.MOVE | Inst.GO, *_) | Inst.RETURN:
                self._pos_num += 1
            case Inst.LASE:
                self._lase_num += 1
        self._pointer += 1

    def plot_xy(self, ax=None):
        """Plot the toolpath in the Cartesian plane with X and Y.

        Args:
            ax: The matplotlib axis to plot on. Will try to a square one by
                default.

        Returns:
            The axis ax, whichever was plotted on.
        """
        if ax is None:
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')

        # These represent the current state, and if we're finished, there's
        #  special behavior.
        pos_num = self._pos_num if self._pos_num != len(self._positions) else -1
        lase_num = self._lase_num if self._lase_num != len(self._lases) else -1

        pos_x, pos_y, _, _, _, _ = zip(*self._positions[pos_num:])
        lase_x, lase_y, _, _, _, _ = zip(*self._lases)

        # Change from um to m
        pos_x = np.array(pos_x) / 1e6
        pos_y = np.array(pos_y) / 1e6
        lase_x = np.array(lase_x) / 1e6
        lase_y = np.array(lase_y) / 1e6

        # Plot the positions, the actual tool's path, as a line
        ax.plot(
            pos_x, pos_y,
        )

        # Plot the lases as a hexagons
        lase_colors = ['yellowgreen'] * self._lase_num + \
                      ['orange'] * (len(self._lases) - self._lase_num)
        lase_colors[0] = 'green'
        lase_colors[-1] = 'orangered'
        lase_colors[lase_num] = 'yellow'

        ax.scatter(
            lase_x, lase_y,
            s=400,
            c=lase_colors,
            marker='H',
        )

        ax.xaxis.set_major_formatter(ticker.EngFormatter(unit='m'))
        ax.yaxis.set_major_formatter(ticker.EngFormatter(unit='m'))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        return ax


def toolpath(inst_gen_function: Generator):
    """A decorator to apply to a generator of instructions. The decorated
    generator will generate a special toolpath iterator, which has utilities
    for plotting, visualizing, error-checking, and will be automatically
    simplified.

    The generator should yield instructions of the formats:
    - (Inst.MOVE, X (float, required), Y (optional), Z, N, P, V)
    - (Inst.GO, X (float, required), Y (optional), Z, N, P, V)
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
