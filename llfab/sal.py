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
    Z_UNIT = (403.4, 'mm')
    N_UNIT = (0, 'deg')
    P_UNIT = (-100, 'deg')
    V_UNIT = (89, 'deg')

    P_POLE_HEIGHT_MM = 111  # mm
    N_ARM_LENGTH_MM = 74  # mm
    P_ARM_LENGTH_MM = 51  # mm
    V_ARM_LENGTH_MM = 74  # mm (Should be equal to N_ARM_LENGTH_MM)
    Z_DEFAULT_MM = 120  # mm (Should be greater than P_POLE_HEIGHT_MM)

    assert N_ARM_LENGTH_MM == V_ARM_LENGTH_MM, 'These have to be equal, it ' \
        'would matter if we were using the N-Axis, but right now we ' \
        'basically want to keep x=0.'
    assert Z_DEFAULT_MM > P_POLE_HEIGHT_MM, 'If the default z position for ' \
        'the lenslet is less than the height of the P_Axis\'s pole, then at ' \
        'incline=90deg, the z-axis won\'t be able to go low enough'

    def __init__(self):
        # Capture all six motors
        self.x = motor.MotorController(serial=self.X_SERIAL, unit=self.X_UNIT)
        self.y = motor.MotorController(serial=self.Y_SERIAL, unit=self.Y_UNIT)
        self.z = motor.MotorController(serial=self.Z_SERIAL, unit=self.Z_UNIT)
        self.n = motor.MotorController(serial=self.N_SERIAL, unit=self.N_UNIT)
        self.p = motor.MotorController(serial=self.P_SERIAL, unit=self.P_UNIT)
        self.v = motor.MotorController(serial=self.V_SERIAL, unit=self.V_UNIT)
        # TODO: Set up bounds

    def get_position(self):
        """Return a tuple of all the motors' positions."""
        return self.x.get_position(), self.y.get_position(), \
            self.z.get_position(), self.n.get_position(), \
            self.p.get_position(), self.v.get_position(),

    def set_zero(self):
        """Set the current position as zero."""
        self.x.set_zero()
        self.y.set_zero()
        self.z.set_zero()
        self.n.set_zero()
        self.p.set_zero()
        self.v.set_zero()

    def to_pos(self, azimuth: float, incline: float, height=Z_DEFAULT_MM):
        """TODO

        """
        x_mm = self.N_ARM_LENGTH_MM - self.V_ARM_LENGTH_MM
        y_mm = -math.sin(incline) * self.P_ARM_LENGTH_MM
        z_mm = height - self.P_POLE_HEIGHT_MM + math.cos(incline) * \
            self.P_ARM_LENGTH_MM
        n_deg = 0
        p_deg = incline
        v_deg = azimuth

        n_hat = np.asarray([-math.cos(n_deg), math.sin(n_deg), 0])
        p_hat = np.asarray([-math.sin(n_deg), math.cos(n_deg), 0])
        z_hat = np.asarray([0, 0, -1])

        n_pos = np.asarray([x_mm, y_mm, 0])
        p_pos = n_pos + self.N_ARM_LENGTH_MM * n_hat + \
            (z_mm + self.P_POLE_HEIGHT_MM) * z_hat
        o_pos = (p_pos + self.P_ARM_LENGTH_MM * (-math.cos(p_deg) * z_hat +
                                                 math.sin(p_deg) * p_hat) -
                 self.V_ARM_LENGTH_MM * n_hat)
        _logger.debug(f'Moving origin to ({int(o_pos[0])}mm, '
                      f'{int(o_pos[1])}mm, {int(o_pos[2])}mm)')

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