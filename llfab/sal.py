"""

"""

from typing import Iterable, TypeAlias

import enum
import logging
import math

from llfab import ezcad
from llfab import motor


_logger = logging.getLogger(__name__)


class Motors(enum.Enum):
    X = 'x'
    Y = 'y'
    Z = 'z'
    N = 'n'
    P = 'p'
    V = 'v'
    ALL = 'all'


class SixAxisLaserController:
    """
    Class to represent our laser setup. SAL for short.

    TODO(Andrew): Turn this brainstorm into documentation:
    the thingy that you invoke from the command line
    the interface with stuff specifically tailored to _our_ tasks and _our_
        setup. basically, the space for andrew's functions
    it would _not_ have a function to do the grid thing, as the grid thing
        is a whole workflow. that would go in scripts, or if it was large,
        would have a subpackage of its own.

    this should represent our machine, the thing sitting to my right
        I should name it after the machine, but first i gotta name the
        machine
    it would also control the laser
        it probably wouldn't need import pyximc at all, it would just call
        the easy functions i made in MotorController

    __init__():

        self.x = capture_motor(serial=17582)
        self.y = capture_motor(...)
        ...

    Other notes:
        this would also be the place to code in any restrictions that the
        firmware can't handle.
    """

    X_SERIAL = 17582
    Y_SERIAL = 17570
    Z_SERIAL = 17631
    N_SERIAL = 17615
    P_SERIAL = 17638
    V_SERIAL = 17607

    X_UNIT = (100, 'mm')
    Y_UNIT = (100, 'mm')
    Z_UNIT = (399.40625, 'mm')
    N_UNIT = (99.88, 'deg')
    P_UNIT = (-100.4, 'deg')
    V_UNIT = (-99.87, 'deg')

    Z_BOUNDS = (0, 45)  # TODO

    P_RADIUS_MM = 56.5  # Compare the height of the origin at p=0 and 90 deg.
    Z_ZERO_MM = 105  # Define z=0 when the v-bed is 105 mm off of the xy bed
    # Note: The focal point of the laser is at around 150 mm off the xy bed.
    Z_DEFAULT_MM = 150  # The height which it will try to keep the origin at.
    N_DEFAULT_DEG = -45  # We don't need the n-axis to move, so fix it.

    # assert Z_DEFAULT_MM > Z_ZERO_MM + P_RADIUS_MM, 'If the default z ' \
    #     'position for the lenslet is less than the height of the P_Axis\'s ' \
    #     'pole, then at incline=90deg, the z-axis won\'t be able to go low ' \
    #     'enough'

    _instance = None

    def __new__(cls, *args, **kwargs):
        new_instance = super().__new__(cls)

        # Specifically use SixAxisLaserController._instance instead of
        #  cls._instances because subclassing this doesn't double the number of
        #  motors available for capture.
        if SixAxisLaserController._instance is not None:
            _logger.debug('Freeing previous instance of SALC.')
            SixAxisLaserController._instance.free()
        SixAxisLaserController._instance = new_instance
        return new_instance

    def __init__(self, capture: Iterable[Motors] | Motors = Motors.ALL):
        # Declare (guaranteed) interface
        self.motors: tuple[motor.MotorController]
        self._free: bool = True  # If I am ._free, I don't need to be .free()d.
        self._required_motors: tuple[Motors]
        self._six_motors: tuple[motor.MotorController | None]

        # Set self._required to capture, or all the motors, if it's Motors.ALL
        self._required_motors = tuple(capture) if capture != Motors.ALL else (
            Motors.X, Motors.Y, Motors.Z, Motors.N, Motors.P, Motors.V,
        )

        # Capture the motors promised by self.captured
        # I know this is a long block, but I did it explicitly instead of with
        #  a loop because it's not iteration. It should be casework.
        #  If you were to make it a loop, you'd be introducing the idea of a
        #  'next' motor and that's not really intuitive.
        self.motors = ()
        self._free = False  # .free() can be called once self.motors exists.
        if Motors.X in self._required_motors:
            self.x = motor.MotorController(
                serial=self.X_SERIAL,
                unit=self.X_UNIT,
            )
            self.motors += (self.x, )
        if Motors.Y in self._required_motors:
            self.y = motor.MotorController(
                serial=self.Y_SERIAL,
                unit=self.Y_UNIT,
            )
            self.motors += (self.y, )
        if Motors.Z in self._required_motors:
            self.z = motor.MotorController(
                serial=self.Z_SERIAL,
                unit=self.Z_UNIT,
            )
            self.motors += (self.z, )
        if Motors.N in self._required_motors:
            self.n = motor.MotorController(
                serial=self.N_SERIAL,
                unit=self.N_UNIT,
            )
            self.motors += (self.n, )
        if Motors.P in self._required_motors:
            self.p = motor.MotorController(
                serial=self.P_SERIAL,
                unit=self.P_UNIT,
            )
            self.motors += (self.p, )
        if Motors.V in self._required_motors:
            self.v = motor.MotorController(
                serial=self.V_SERIAL,
                unit=self.V_UNIT,
            )
            self.motors += (self.v, )

        # Make self._six_motors, an internal variable so I can deal with
        #  situations where I don't need all six motors, but I do need the
        #  order that they're normally in.
        self._six_motors = (getattr(self, m) if hasattr(self, m) else None
                            for m in ('x', 'y', 'z', 'n', 'p', 'v'))

        # TODO: Set up bounds

    def _check_captured(self, *motors: Motors):
        """Returns True if all of the specified motors were required at
        at instantiation and raise a MissingMotorError otherwise"""
        if Motors.ALL in motors:
            motors = [
                Motors.X, Motors.Y, Motors.Z, Motors.N, Motors.P, Motors.V,
            ]

        if all(m in self._required_motors for m in motors):
            return True

        raise MissingMotorError(
            need=motors,
            captured=self._required_motors,
        )

    @classmethod
    def calc_origin(cls, x_mm, y_mm, z_mm, n_deg, p_deg, v_deg):
        """Calculates the expected location of the origin given positions."""
        n_rad = math.radians(n_deg)
        p_rad = math.radians(p_deg)
        # v_rad = math.radians(v_deg)

        return (
            x_mm - cls.P_RADIUS_MM * math.sin(p_rad) * math.sin(n_rad),
            y_mm + cls.P_RADIUS_MM * math.sin(p_rad) * math.cos(n_rad),
            z_mm + cls.Z_ZERO_MM + cls.P_RADIUS_MM * (1 - math.cos(p_rad)),
        )

    def free(self):
        """Free all captured motors."""
        if self._free:
            _logger.debug('SALC.motors does not exist, nothing to free.')
            return
        for mot in self.motors:
            mot.free()
        self._free = True

    def get_origin(self):
        """Return the (expected) current location of the origin in XYZ."""
        return self.calc_origin(*self.get_position())

    def get_position(self) -> tuple[float]:
        """Return a tuple of all the motors' positions. Defaults to 0 for
        non-captured motors."""
        return tuple(float(m.get_position()) if m is not None else 0.0
                     for m in self._six_motors)

    def get_position_step(self) -> tuple[int]:
        """Return a tuple of all the motors' positions in steps. Defaults to
        0 for non-captured motors."""
        return tuple(int(m.get_position_step()[0]) if m is not None else 0
                     for m in self._six_motors)

    def get_position_microstep(self) -> tuple[int]:
        """Return a tuple of all the motors' positions in steps. Defaults to
        0 for non-captured motors."""
        return tuple(int(m.get_position_step()[1]) if m is not None else 0
                     for m in self._six_motors)

    def set_zero(self):
        """Set the current position as zero."""
        _logger.info('Defining current 6-axis position as zero.')
        for mot in self.motors:
            mot.set_zero()

    def to_spherical_pos(
            self,
            azimuth: float,
            incline: float,
            height: float = Z_DEFAULT_MM,
    ):
        """Moves the 6-axis to the given spherical coordinates, in degrees.
        Requires all six motors to be captured."""
        self._check_captured(Motors.ALL)

        # The angles are independent, and we calculate the xyz.
        n_deg = self.N_DEFAULT_DEG
        p_deg = incline
        v_deg = azimuth

        # Get radians for math.sin & math.cos functions
        n_rad = math.radians(n_deg)
        p_rad = math.radians(incline)
        # v_rad = math.radians(azimuth)

        # Calculate where each stepper should go to
        x_mm = (self.P_RADIUS_MM * math.sin(p_rad) * math.sin(n_rad))
        y_mm = (-self.P_RADIUS_MM * math.sin(p_rad) * math.cos(n_rad))
        z_mm = (height - self.Z_ZERO_MM - self.P_RADIUS_MM *
                (1 - math.cos(p_rad)))

        o_pos = self.calc_origin(x_mm, y_mm, z_mm, n_deg, p_deg, v_deg)
        _logger.debug(f'Moving origin to ({int(o_pos[0])}mm, '
                      f'{int(o_pos[1])}mm, {int(o_pos[2])}mm)')

        # Actually move it.
        self.x.move_to(unit=x_mm)
        self.y.move_to(unit=y_mm)
        self.z.move_to(unit=z_mm)
        self.n.move_to(unit=n_deg)
        self.p.move_to(unit=p_deg)
        self.v.move_to(unit=v_deg)

    def to_xy_pos(self, x_mm: float, y_mm: float, rail: bool = False):
        """Moves the 6-axis in X and Y, in mm.

        Requires those two motors. To move step-wise, see
        MotorController.move_to and just access SALC.x and SALC.y manually.

        Args:
            x_mm: The x-position to go to, in millimeters.
            y_mm: The y-position to go to, in millimeters.
            rail: If True, then instead of throwing an error if you go out of
                bounds, it will move as far as allowed.

        Raises:
            MissingMotorError: If you did not capture motors X and Y.
            LibXIMCCommandFailedError: If the movement fails.
            NoUserUnitError: If you supply unit but don't define user units.
            PositionOutOfBoundsError: If rail=False and you supply a position
                that's out of bounds.
        """
        self._check_captured(Motors.X, Motors.Y)

        self.x.move_to(unit=x_mm, rail=rail)
        self.y.move_to(unit=y_mm, rail=rail)

    def to_zero(self):
        """Move all motors to their zero position."""
        _logger.debug(f'Moving all motors to zero.')
        for mot in self.motors:
            mot.move_to(0)

    @staticmethod
    def lase():
        """Run the laser. Currently just presses F2 on EZCAD."""
        ezcad.ezcad_lase()

    def __del__(self):
        self.free()


# The name's too long
SALC: TypeAlias = SixAxisLaserController


class MissingMotorError(Exception):
    """Raised when a function is called that requires motors which SALC did not
    capture."""

    DEFAULT_MSG = 'Motors {0} required to use this function, missing {1}.'

    def __init__(self,
                 need: Iterable[Motors],
                 captured: Iterable[Motors],
                 message: str = DEFAULT_MSG):
        need_msg = ''.join(f'{str(m.value).upper()}, ' for m in need)[:-2]
        missing_msg = ''.join(f'{str(m.value).upper()}, ' for m in need
                              if m not in captured)[:-2]
        super().__init__(message.format(need_msg, missing_msg))
