"""

"""

from typing import Iterable, TypeAlias, Optional

import asyncio
import enum
import datetime
import logging
import math
import time

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

from llfab import ezcad
from llfab import gas
from llfab import motor
from llfab import config


_logger = logging.getLogger(__name__)


class Motors(enum.Enum):
    X = 'x'
    Y = 'y'
    Z = 'z'
    N = 'n'
    P = 'p'
    V = 'v'
    ALL = 'all'


class Inst(enum.Enum):
    MOVE = 'MOVE'
    GO = 'GO'
    GO_SPH = 'GO_SPH'
    LASE = 'LASE'
    FENCE = 'FENCE'
    RETURN = 'RETURN'


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

    X_SERIAL = config['serial']['x']
    Y_SERIAL = config['serial']['y']
    Z_SERIAL = config['serial']['z']
    N_SERIAL = config['serial']['n']
    P_SERIAL = config['serial']['p']
    V_SERIAL = config['serial']['v']

    X_UNIT = config['unit']['x']
    Y_UNIT = config['unit']['y']
    Z_UNIT = config['unit']['z']
    N_UNIT = config['unit']['n']
    P_UNIT = config['unit']['p']
    V_UNIT = config['unit']['v']

    X_REFRESH_INTERVAL_MS = config['refresh_interval_ms']['x']
    Y_REFRESH_INTERVAL_MS = config['refresh_interval_ms']['y']
    Z_REFRESH_INTERVAL_MS = config['refresh_interval_ms']['z']
    N_REFRESH_INTERVAL_MS = config['refresh_interval_ms']['n']
    P_REFRESH_INTERVAL_MS = config['refresh_interval_ms']['p']
    V_REFRESH_INTERVAL_MS = config['refresh_interval_ms']['v']

    # Compare the height of the origin at p=0 and 90 deg.
    P_RADIUS_UM = float(config['geometry']['p_radius_um'])

    # Define z=0 when the v-bed is 105 um off of the xy bed
    # TODO: This is not being used
    # Z_ZERO_UM = float(config['geometry']['z_zero_um'])

    # The height which it will try to keep the origin at.
    # Note: The focal point of the laser is at around 150 um off the xy bed.
    # TODO: This is not being used
    # Z_DEFAULT_UM = float(config['geometry']['z_default_um'])

    # We don't need the n-axis to move, so fix it in this position.
    N_DEFAULT_DEG = float(config['geometry']['n_default_deg'])

    # assert Z_DEFAULT_UM > Z_ZERO_UM + P_RADIUS_UM, 'If the default z ' \
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
                refresh_interval_ms=self.X_REFRESH_INTERVAL_MS,
            )
            self.motors += (self.x, )
        if Motors.Y in self._required_motors:
            self.y = motor.MotorController(
                serial=self.Y_SERIAL,
                unit=self.Y_UNIT,
                refresh_interval_ms=self.Y_REFRESH_INTERVAL_MS,
            )
            self.motors += (self.y, )
        if Motors.Z in self._required_motors:
            self.z = motor.MotorController(
                serial=self.Z_SERIAL,
                unit=self.Z_UNIT,
                refresh_interval_ms=self.Z_REFRESH_INTERVAL_MS,
            )
            self.motors += (self.z, )
        if Motors.N in self._required_motors:
            self.n = motor.MotorController(
                serial=self.N_SERIAL,
                unit=self.N_UNIT,
                refresh_interval_ms=self.N_REFRESH_INTERVAL_MS,
            )
            self.motors += (self.n, )
        if Motors.P in self._required_motors:
            self.p = motor.MotorController(
                serial=self.P_SERIAL,
                unit=self.P_UNIT,
                refresh_interval_ms=self.P_REFRESH_INTERVAL_MS,
            )
            self.motors += (self.p, )
        if Motors.V in self._required_motors:
            self.v = motor.MotorController(
                serial=self.V_SERIAL,
                unit=self.V_UNIT,
                refresh_interval_ms=self.V_REFRESH_INTERVAL_MS,
            )
            self.motors += (self.v, )

        # Make self._six_motors, an internal variable so I can deal with
        #  situations where I don't need all six motors, but I do need the
        #  order that they're normally in.
        self._six_motors = tuple(getattr(self, m) if hasattr(self, m) else None
                                 for m in ('x', 'y', 'z', 'n', 'p', 'v'))

    # --- Setup & Teardown ----------------------------------------------------
    def free(self):
        """Free all captured motors."""
        if self._free:
            _logger.debug('SALC.motors does not exist, nothing to free.')
            return
        for mot in self.motors:
            mot.free()
        self._free = True

    # --- Utility -------------------------------------------------------------
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

    # --- Geometry Calculations -----------------------------------------------
    @classmethod
    def calc_origin(cls, x_um, y_um, z_um, n_deg, p_deg, v_deg):
        """Calculates the expected location of the origin given positions."""
        n_rad = math.radians(n_deg)
        p_rad = math.radians(p_deg)
        # v_rad = math.radians(v_deg)

        return (
            x_um - cls.P_RADIUS_UM * math.sin(p_rad) * math.sin(n_rad),
            y_um + cls.P_RADIUS_UM * math.sin(p_rad) * math.cos(n_rad),
            z_um + cls.P_RADIUS_UM * (1 - math.cos(p_rad)),
        )

    @classmethod
    def calc_spherical_pos(
            cls,
            azimuth: float,
            incline: float,
            height: float,
    ):
        """Gets the 6-coordinates corresponding to a given spherical
        coordinate system."""
        # The angles are independent, and we calculate the xyz.
        n_deg = cls.N_DEFAULT_DEG
        p_deg = incline
        v_deg = azimuth

        # Get radians for math.sin & math.cos functions
        n_rad = math.radians(n_deg)
        p_rad = math.radians(incline)
        # v_rad = math.radians(azimuth)

        # Calculate where each stepper should go to
        x_um = (cls.P_RADIUS_UM * math.sin(p_rad) * math.sin(n_rad))
        y_um = (-cls.P_RADIUS_UM * math.sin(p_rad) * math.cos(n_rad))
        z_um = (height - cls.P_RADIUS_UM * (1 - math.cos(p_rad)))

        return x_um, y_um, z_um, n_deg, p_deg, v_deg

    # --- Information (Getters) -----------------------------------------------
    def at(self,
           x_um: Optional[float] = None,
           y_um: Optional[float] = None,
           z_um: Optional[float] = None,
           n_deg: Optional[float] = None,
           p_deg: Optional[float] = None,
           v_deg: Optional[float] = None) -> bool:
        """Check if the arm is at the given coordinates.

        All coordinates are optional. If you give any value for a motor that
        you don't have captured, it will error.

        Args:
            x_um: The x-coordinate, in micrometers.
            y_um: The y-coordinate, in micrometers.
            z_um: The z-coordinate, in micrometers.
            n_deg: The n-coordinate, in degrees.
            p_deg: The p-coordinate, in degrees.
            v_deg: The v-coordinate, in degrees.

        Raises:
            MissingMotorError: If you try to check a motor you haven't captured.

        Returns: True if the motors are within a microstep of the given
        coordinates, False otherwise.
        """
        all_motors = [Motors.X, Motors.Y, Motors.Z, Motors.N, Motors.P,
                      Motors.V]
        units = [self.X_UNIT, self.Y_UNIT, self.Z_UNIT, self.N_UNIT,
                 self.P_UNIT, self.V_UNIT]
        coords = (x_um, y_um, z_um, n_deg, p_deg, v_deg)

        # Check that we have captured all the motors we're asking about.
        required = [mot for index, mot in enumerate(all_motors)
                    if coords[index] is not None]
        self._check_captured(*required)

        # Get the current position
        pos = self.get_position()

        # For each coordinate, check if we're within a microstep.
        #  Note: we check within-a-microstep because this function is meant to
        #  be given in user units, as hence the supplied coordinates might be
        #  between microsteps.
        for coord, ax_pos, (scale, _) in zip(coords, pos, units):
            if coord is None: continue

            # We want units per microstep, as this is going to be the
            #  acceptable rounding error. Scale is (steps / unit), so we will
            #  get (units / microstep) via
            #  1 / ((steps / unit) * (microsteps / step))
            error = 1 / (scale * 256)
            if not (ax_pos - error < coord < ax_pos + error):
                return False
        return True

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

    # --- Configuration (Setters) ---------------------------------------------
    def set_zero(self):
        """Set the current position as zero."""
        motor_names = [m.name for m in self.motors]
        if input(f'This will reset the motors {motor_names} to zero, are you '
                 f'sure you want to do this? (Y/N): ').lower() != 'y':
            return
        _logger.info('Defining current 6-axis position as zero.')
        for mot in self.motors:
            mot.set_zero()

    # --- Movement Functions --------------------------------------------------
    async def to_pos_soft(self, x_um: float, y_um: float, z_um: float,
                          n_deg: float, p_deg: float, v_deg: float,
                          rail: bool = False):
        """Moves the 6-axis to the specified coordinate, ignoring motors that
        are not captured.

        Args:
            x_um: The x-position to go to, in micrometers.
            y_um: The y-position to go to, in micrometers.
            z_um: The z-position to go to, in micrometers.
            n_deg: The n-position to go to, in degrees.
            p_deg: The n-position to go to, in degrees.
            v_deg: The n-position to go to, in degrees.
            rail: If True, then instead of throwing an error if you go out of
                bounds, it will move as far as allowed.

        Raises:
            LibXIMCCommandFailedError: If the movement fails.
            PositionOutOfBoundsError: If rail=False and you supply a position
                that's out of bounds.
        """
        coords = [x_um, y_um, z_um, n_deg, p_deg, v_deg]
        tasks = set()
        for mot, coord in zip(self._six_motors, coords):
            if mot is not None:
                tasks.add(mot.move_to_async(unit=coord, rail=rail))
        await asyncio.gather(*tasks)

    async def to_pos_sph(
            self,
            azimuth: float,
            incline: float,
            height: float,
    ):
        """Moves the 6-axis to the given spherical coordinates, in degrees.
        Requires all six motors to be captured except N."""
        self._check_captured(Motors.X, Motors.Y, Motors.Z, Motors.P, Motors.V)

        x_um, y_um, z_um, n_deg, p_deg, v_deg = self.calc_spherical_pos(
            azimuth=azimuth, incline=incline, height=height,
        )

        o_pos = self.calc_origin(x_um, y_um, z_um, n_deg, p_deg, v_deg)
        _logger.debug(f'Moving origin to ({int(o_pos[0])}um, '
                      f'{int(o_pos[1])}um, {int(o_pos[2])}um)')

        # Actually move it.
        await asyncio.gather(
            self.x.move_to_async(unit=x_um),
            self.y.move_to_async(unit=y_um),
            self.z.move_to_async(unit=z_um),
            # self.n.move_to_async(unit=n_deg),
            self.p.move_to_async(unit=p_deg),
            self.v.move_to_async(unit=v_deg),
        )

    async def to_pos_xy(self, x_um: float, y_um: float, rail: bool = False):
        """Moves the 6-axis in X and Y, in um.

        Requires those two motors. To move step-wise, see
        MotorController.move_to and just access SALC.x and SALC.y manually.

        Args:
            x_um: The x-position to go to, in micrometers.
            y_um: The y-position to go to, in micrometers.
            rail: If True, then instead of throwing an error if you go out of
                bounds, it will move as far as allowed.

        Raises:
            MissingMotorError: If you did not capture motors X and Y.
            LibXIMCCommandFailedError: If the movement fails.
            PositionOutOfBoundsError: If rail=False and you supply a position
                that's out of bounds.
        """
        self._check_captured(Motors.X, Motors.Y)

        await asyncio.gather(
            self.x.move_to_async(unit=x_um, rail=rail),
            self.y.move_to_async(unit=y_um, rail=rail),
        )

    async def to_zero(self):
        """Move all motors to their zero position."""
        _logger.debug(f'Moving all motors to zero.')
        tasks = set()
        for mot in self.motors:
            tasks.add(mot.move_to(0))
        await asyncio.gather(*tasks)

    # --- Toolpath Runner -----------------------------------------------------
    async def run_toolpath(
            self,
            toolpath,
            origin,

            do_lase: bool,
            lase_time_ms: float,
            move_timeout_s: float = 5.0,
            net_timeout_s: float = 10.0,

            plot_type: Optional[str] = None,
            plot_args: Optional[dict] = None,
    ):
        """Runs the supplied toolpath.

        Args:
            toolpath: The toolpath to perform. Don't re-create it each time you
                run this function, as they track progress, and you can continue
                where you left off.
            origin: The origin that you want the toolpath run relative to.
                Check it with SALC.get_position() at the same time you make
                your toolpath, and don't lose track of it!
            do_lase: bool, if True, will actually try to run the laser.
                if False, will simply wait. Useful for testing.
            lase_time_ms: float, in milliseconds, the amount of time it takes
                the laser to run. It will wait exactly this long each time, so
                get it as close as you can.
            move_timeout_s: float, in seconds, the amount of time that
                the computer will wait before deciding that a movement failed.
                Lowering it will *not* speed up anything, it will only cause it
                to break when a particularly long movement occurs.
            net_timeout_s: float, in seconds, the amount of time to wait before
                deciding that there is no internet access.
            plot_type: str, one of the following options:
                - 'xy', for an xy plot, viewed from the top.
                - 'sph', for a spherical plot, meant for lenslets.
            plot_args: dict, arguments for toolpath.plot. See harness.py for
                more info.
        """
        # --- Sanitize input ---
        origin = np.array(origin)
        plot_args = plot_args or {}

        # --- Make async helper functions ---
        # This seems like a roundabout way to do this, but
        #  - Instructions do toolpath.confirm() ASAP and iff they're done
        #  - This allows for much more verbose errors
        #  - The main instruction loop is easier to read
        #  and I think that the 2nd-order functions are worth that.
        async def movement(to_pos):
            print(f'Moving to ({tuple(to_pos)}).')
            try:
                await asyncio.wait_for(
                    self.to_pos_soft(*to_pos),
                    timeout=move_timeout_s,
                )
                toolpath.confirm()
            except asyncio.TimeoutError:
                raise MovementTimeoutError(
                    f'Movement timed out, at position '
                    f'{self.get_position()}, not {to_pos}.)'
                )

        async def lasing():
            print(f'Lasing at {self.get_position()}.')
            if do_lase:
                with self.gas():
                    self.lase()
                    await asyncio.sleep(lase_time_ms / 1000)
            else:
                await asyncio.sleep(lase_time_ms / 1000)
            toolpath.confirm()

        async def ping():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get('http://google.com',
                                           timeout=net_timeout_s) as response:
                        if response.status != 200:
                            raise Exception(
                                f'Connecting to Google, got response '
                                f'#{response.status}, expected 200.')
            except Exception:
                raise NoInternetConnectionError(
                    'Cannot connect to Google, operation suspended. See '
                    'traceback for more details.')

        # --- Check the bounds ---
        positions = list(zip(*toolpath.positions))
        for index, mot in enumerate(self._six_motors):
            if mot is None:
                continue
            lower, upper = mot.get_boundaries()
            for relative_pos in positions[index]:
                if not lower < relative_pos + origin[index] < upper:
                    raise motor.PositionOutOfBoundsError(
                        f'Motor {mot.name}, '
                        f'Position {relative_pos + origin[index]:.2f}'
                    )

        # --- Run the toolpath ---
        instruction_task = None
        connection_task = asyncio.create_task(ping())
        for instruction in toolpath:
            display.clear_output(wait=True)

            # --- Check internet connection ---
            # Do this before there's a risk of running any instructions
            # TODO: Perhaps, to save time, I should only await this before
            #  each lase, as that's the fire hazard step
            await connection_task
            connection_task = asyncio.create_task(ping())

            # --- Dispatch instruction task ---
            # This block should go first, as it takes the longest to run.
            #  Asynchronously, this lets everything else run in the meantime.
            match instruction:
                case (Inst.GO, *to_pos_relative):
                    # Calculate absolute position to go to
                    to_pos_relative = np.array(to_pos_relative)
                    to_pos_relative.resize((6, ))
                    to_pos = origin + to_pos_relative

                    # Schedule the movement task
                    instruction_task = asyncio.create_task(movement(to_pos))

                case Inst.LASE:
                    # Schedule the lasing task
                    instruction_task = asyncio.create_task(lasing())

            # --- Calculate & Print Duration ---
            lases_done, lases_total = toolpath.lase_status
            moves_done, moves_total = toolpath.move_status
            lases_left = lases_total - lases_done
            moves_left = moves_total - moves_done
            duration_s = lase_time_ms * lases_left + 0.5 * moves_left
            finish_time = datetime.datetime.now() + \
                datetime.timedelta(seconds=duration_s)
            print(f'Projected finish time: {finish_time}.')

            # --- Plot the Progress ---
            # TODO: Make this happen rarely, or optionally, as it's expensive
            #  maybe just when it's lasing ?
            toolpath.plot(kind=plot_type, **plot_args)
            plt.show()

            # This ensures that the instructions go in order, and that
            #  toolpath.confirm() is called before the next instruction is
            #  pulled from the list.
            await instruction_task

        # --- Finished Routine ---
        print('Done!')
        # TODO: Send an email to Shawn

    # --- Static Controller Functions -----------------------------------------
    # These functions forward to similar functions in other files, I have them
    #  here for centralized control.
    @staticmethod
    def lase():
        """Run the laser. Currently just presses F2 on EZCAD."""
        ezcad.ezcad_lase()

    @staticmethod
    def gas():
        """Return a context which guarantees that the gas will at least
        attempt to shut off."""
        return gas.gas()

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


class MovementTimeoutError(Exception):
    """Raised when moving takes too long or is expected to never complete."""


class NoInternetConnectionError(Exception):
    """Raised when internet connection fails during toolpath execution."""
    pass
