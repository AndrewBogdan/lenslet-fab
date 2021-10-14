"""

"""

from llfab import motor


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

    def __init__(self):
        self.x = motor.MotorController(serial=self.X_SERIAL)
        self.y = motor.MotorController(serial=self.Y_SERIAL)
        self.z = motor.MotorController(serial=self.Z_SERIAL)
        self.n = motor.MotorController(serial=self.N_SERIAL)
        self.p = motor.MotorController(serial=self.P_SERIAL)
        self.v = motor.MotorController(serial=self.V_SERIAL)

    # TODO: Make the __del__ so that when the class is destroyed, it relases
    #  the motors.
