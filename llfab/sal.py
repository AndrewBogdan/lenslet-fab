"""

"""

import logging
import math

import numpy as np

from llfab import motor


_logger = logging.getLogger(__name__)


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
    Z_DEFAULT_MM = 150  # The height which it will try to keep the origin at.
    N_DEFAULT_DEG = -45  # We don't need the n-axis to move, so fix it.

    # assert Z_DEFAULT_MM > Z_ZERO_MM + P_RADIUS_MM, 'If the default z ' \
    #     'position for the lenslet is less than the height of the P_Axis\'s ' \
    #     'pole, then at incline=90deg, the z-axis won\'t be able to go low ' \
    #     'enough'

    def __init__(self):
        # Capture all six motors
        self.x = motor.MotorController(serial=self.X_SERIAL, unit=self.X_UNIT)
        self.y = motor.MotorController(serial=self.Y_SERIAL, unit=self.Y_UNIT)
        self.z = motor.MotorController(serial=self.Z_SERIAL, unit=self.Z_UNIT)
        self.n = motor.MotorController(serial=self.N_SERIAL, unit=self.N_UNIT)
        self.p = motor.MotorController(serial=self.P_SERIAL, unit=self.P_UNIT)
        self.v = motor.MotorController(serial=self.V_SERIAL, unit=self.V_UNIT)
        self.motors = [self.x, self.y, self.z, self.n, self.p, self.v]
        # TODO: Set up bounds

    def get_origin(self):
        """Return the (expected) current location of the origin."""
        return self.calc_origin(*self.get_position())

    def calc_origin(self, x_mm, y_mm, z_mm, n_deg, p_deg, v_deg):
        """Calculates the expected location of the origin given positions."""
        n_rad = math.radians(n_deg)
        p_rad = math.radians(p_deg)
        # v_rad = math.radians(v_deg)

        return (
            x_mm - self.P_RADIUS_MM * math.sin(p_rad) * math.sin(n_rad),
            y_mm + self.P_RADIUS_MM * math.sin(p_rad) * math.cos(n_rad),
            z_mm + self.Z_ZERO_MM + self.P_RADIUS_MM * (1 - math.cos(p_rad)),
        )

    def get_position(self):
        """Return a tuple of all the motors' positions."""
        return [m.get_position() for m in self.motors]

    def get_position_step(self):
        """Return a tuple of all the motors' positions in steps."""
        return [m.get_position_step()[0] for m in self.motors]

    def get_position_microstep(self):
        """Return a tuple of all the motors' positions in steps."""
        return [m.get_position_step()[1] for m in self.motors]

    def set_zero(self):
        """Set the current position as zero."""
        _logger.info('Defining current 6-axis position as zero.')
        self.x.set_zero()
        self.y.set_zero()
        self.z.set_zero()
        self.n.set_zero()
        self.p.set_zero()
        self.v.set_zero()

    def to_pos(self, azimuth: float, incline: float, height=Z_DEFAULT_MM):
        """TODO

        """
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
        self.x.move_to(x_mm)
        self.y.move_to(y_mm)
        self.z.move_to(z_mm)
        self.n.move_to(n_deg)
        self.p.move_to(p_deg)
        self.v.move_to(v_deg)

    def to_zero(self):
        """Move all motors to their zero position."""
        _logger.debug(f'Moving all motors to zero.')
        self.x.move_to(0)
        self.y.move_to(0)
        self.z.move_to(0)
        self.n.move_to(0)
        self.p.move_to(0)
        self.v.move_to(0)

    # TODO: Make the __del__ so that when the class is destroyed, it releases
    #  the motors.


# The name's too long
SALC = SixAxisLaserController
