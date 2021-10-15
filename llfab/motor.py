"""
Module for controlling the motors through pyximc.
"""

import logging

import pyximc
from pyximc import lib as libximc

_logger = logging.getLogger(__name__)


class MotorController:
    """
    class MotorController()
        the datatype of machine.x, y, z, etc
        call it motor, controller, or axes
    """

    def __init__(self, serial: int):
        self._capture_device_by_serial(serial)

    def _capture_device_by_serial(self, serial: int):
        """Captures the motor specified by self.serial.

        Args:
            serial (int): The device to capture's serial number.
        """

        # We make a probe to enumerate all the devices we have access to.
        probe_flags = pyximc.EnumerateFlags.ENUMERATE_PROBE
        # If we were searching for network devices, we would use
        #  EnumerateFlags.ENUMERATE_NETWORK and hint b"addr=<url>". See docs
        #  for the libximc.enumerate_devices function.
        device_enum = libximc.enumerate_devices(probe_flags, '')

        # Count the number of devices so we cn iterate over it.
        device_count = libximc.get_device_count(device_enum)
        _logger.debug("Device count: " + repr(device_count))

        # Make a empty spaces to store information, which we'll
        #  pass-by-reference into.
        controller_name_struct = pyximc.controller_name_t()
        serial_num_struct = pyximc.c_uint()
        # Iterate over all devices, finding their friendly controller name.
        for device_index in range(0, device_count):
            # Get the device's COM port name
            port_name = libximc.get_device_name(device_enum, device_index)

            # Get the device's controllers friendly name
            result = libximc.get_enumerate_device_controller_name(
                device_enum,
                device_index,
                # Pass-by-reference to store controller name in controller_name
                pyximc.byref(controller_name_struct)
            )
            if result != pyximc.Result.Ok:
                _logger.debug(f'Failed to retrieve controller name for device '
                              f'#{device_index}, error: {result}')
                continue
            friendly_name = controller_name_struct.ControllerName.decode('utf-8')

            # Get the device's serial number
            result = libximc.get_enumerate_device_serial(
                device_enum,
                device_index,
                pyximc.byref(serial_num_struct)
            )
            if result != pyximc.Result.Ok:
                _logger.debug(f'Failed to retrieve serial number for device '
                              f'#{device_index}, error: {result}')
                continue
            serial_num = serial_num_struct.value

            _logger.debug(
                f'Found device #{device_index}, '
                f'port name: "{port_name.decode("utf-8")}", '
                f'friendly name: "{friendly_name}", '
                f'serial: {serial_num}'
            )

            # Check if it's the serial number we're looking for
            if serial_num == serial:
                _logger.info(f'Opening motor controller "{friendly_name}"')
                device_id = libximc.open_device(port_name)
                _logger.debug(f'Device id: {device_id}')
                # Success! Save information and return.
                self.port = port_name.decode("utf-8")
                self.name = friendly_name
                self.device_id = device_id
                self.serial = serial_num
                return

        # Raise an error because we couldn't find our device
        err_msg = f'Could not find device with serial {serial}'
        _logger.error(err_msg)
        raise DeviceNotFoundError(err_msg)

    def get_position(self):
        """Get the motor's current position.

        Returns: a tuple of:
            step (int): the step number
            microstep (int): the microstep TODO: What range?
        """
        _logger.debug(f'Getting position of {self.name}')
        position_struct = pyximc.get_position_t()
        result = libximc.get_position(
            self.device_id,
            pyximc.byref(position_struct),
        )
        if result != pyximc.Result.Ok:
            raise LibXIMCCommandFailedError()
        step = position_struct.Position  # CType uint
        microstep = position_struct.uPosition  # CType uint
        # encoder_pos = position_struct.EncPosition  # CType long
        return step, microstep

    def move_to(self, step: int, microstep: int = 0):
        """
        Move to the specified location.

        Args:
            step: The step to move to
            microstep (int): The microstep within that step, # TODO: Range?

        Raises:
            LibXIMCCommandFailedError: If the movement fails
        """
        _logger.debug(f'Moving {self.name} to step {step} + {microstep}/256')
        result = libximc.command_move(
            self.device_id,
            step,
            microstep,
        )
        if result != pyximc.Result.Ok:
            raise LibXIMCCommandFailedError()

    def move_by(self, step: int, microstep: int = 0):
        """
        Move to the specified number of steps.

        Args:
            step: The number of steps to move by
            microstep (int): The microstep offset, in [-255, 255]

        Raises:
            LibXIMCCommandFailedError: If the movement fails
        """
        _logger.debug(f'Moving {self.name} by {step} + {microstep}/256 steps')
        result = libximc.command_movr(
            self.device_id,
            step,
            microstep,
        )
        if result != pyximc.Result.Ok:
            raise LibXIMCCommandFailedError()


class DeviceNotFoundError(Exception):
    """Raised when the requested device could not be captured."""
    pass


class LibXIMCCommandFailedError(Exception):
    """Raised when a libximc command fails unexpectedly."""
    pass


def preamble():
    """Sets up pyximc to start manipulating motors."""

    # Check the version of the XIMC library
    sbuf = pyximc.create_string_buffer(64)
    libximc.ximc_version(sbuf)
    _logger.info("pyXIMC version: " + sbuf.raw.decode().rstrip("\0"))

    # If we were using networked controllers, here is where you'd set a bindy
    #  key. There's information on it in the pyximc test scripts.
