"""
Module for controlling the motors through pyximc.
"""

from typing import Optional, Tuple
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

    def __init__(self,
                 serial: int,
                 unit: Optional[Tuple[float, str]] = None):
        # --- Instance Variables ---
        #  Device connection information
        self.port: str
        self.name: str
        self.device_id: int
        self.serial: int
        #  Calibration information
        self.unit: Optional[pyximc.calibration_t] = None
        # self.unit_bounds: Optional[Tuple[float, float]] = None
        self.unit_name: Optional[str] = None

        # --- Setup ---
        self._capture_device_by_serial(serial)
        if unit: self._set_user_unit(*unit)

    # --- Initialization ------------------------------------------------------
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
            friendly_name = \
                controller_name_struct.ControllerName.decode('utf-8')

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

    def _set_user_unit(self,
                       steps_per_unit: float,
                       # lower_bound: float,
                       # upper_bound: float,
                       unit_name: str = ''):
        """
        Set the motor's user unit to scale by steps_per_unit.

        Args:
            steps_per_unit (float): The multiplier to convert units to steps
            unit_name (str): The name of the unit, helpful for debugging.
        """
        _logger.debug(f'Setting user unit of motor {self.name} to unit '
                      f'{unit_name}, {steps_per_unit} steps per unit')
        self.unit = pyximc.calibration_t()
        self.unit.A = steps_per_unit
        # self.unit_bounds = (lower_bound, upper_bound)
        self.unit_name = unit_name

    # --- Information (Getters) -----------------------------------------------
    def get_position(self):
        """
        Get the motor's current position in user units.

        Returns:
            position (float): The position in user units.
        """
        step, microstep = self.get_position_step()
        # TODO: Use the libximc for this, not my own math.
        return (step + microstep / 256) * self.unit.A

    def get_position_step(self):
        """
        Get the motor's current position in steps.

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
            raise LibXIMCCommandFailedError(result)
        step = position_struct.Position  # CType uint
        microstep = position_struct.uPosition  # CType uint
        # encoder_pos = position_struct.EncPosition  # CType long
        return step, microstep

    # --- Settings Customization (Setters) ------------------------------------
    def set_zero(self):
        """Set the current position as zero."""
        _logger.debug(f'Defining currently position of {self.name} as zero')
        result = libximc.command_zero(
            self.device_id,
        )
        if result != pyximc.Result.Ok:
            raise LibXIMCCommandFailedError(result)

    # --- Movement Functions --------------------------------------------------
    # TODO: Should I have separate move functions for user units and steps,
    #  or should it be a flag?
    def move_to(self, pos: float):
        """
        Move to the specified location, in user units (absolute positioning).

        Args:
            pos (float): The position to move to, in user units.

        Raises:
            LibXIMCCommandFailedError: If the movement fails
        """
        if not self.unit:
            raise NoUserUnitError()
        # TODO: Check bounds once a zero is defined.
        _logger.debug(f'Moving {self.name} to position {pos} {self.unit_name}')
        self.move_to_step(int((self.unit.A * pos) // 1),
                          int((256 * (self.unit.A * pos)) % 1))
        # result = libximc.command_move_calb(
        #     self.device_id,
        #     pyximc.c_float(pos),
        #     pyximc.byref(self.unit),
        # )
        # result = libximc.command_move(  # TODO: _calb gives me result = -3
        #     self.device_id,
        #     int((self.unit.A * pos) // 1),
        #     int((256 * (self.unit.A * pos)) % 1),
        # )
        # if result != pyximc.Result.Ok:
        #     raise LibXIMCCommandFailedError(result)

    def move_by(self, by: float):
        """
        Move by the specified number of user units (relative positioning).

        Args:
            by (float): The number of user units to move by.

        Raises:
            LibXIMCCommandFailedError: If the movement fails
            NoUserUnitError: If the user never defined user units.
        """
        if not self.unit:
            raise NoUserUnitError()
        # TODO: Check bounds once a zero is defined.
        _logger.debug(f'Moving {self.name} by {by} {self.unit_name}')
        self.move_by_step(int((self.unit.A * by) // 1),
                          int((256 * (self.unit.A * by)) % 1))
        # result = libximc.command_movr_calb(
        #     self.device_id,
        #     pyximc.c_float(by),
        #     pyximc.byref(self.unit),
        # )
        # result = libximc.command_movr(  # TODO: _calb gives me result = -3
        #     self.device_id,
        #     int((self.unit.A * by) // 1),
        #     int((256 * (self.unit.A * by)) % 1),
        # )
        # if result != pyximc.Result.Ok:
        #     raise LibXIMCCommandFailedError(result)

    def move_to_step(self, step: int, microstep: int = 0):
        """
        Move to the specified location, in steps (absolute positioning).

        Args:
            step (int): The step to move to
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
            raise LibXIMCCommandFailedError(result)

    def move_by_step(self, step: int, microstep: int = 0):
        """
        Move by the specified number of steps (relative positioning).

        Args:
            step (int): The number of steps to move by
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
            raise LibXIMCCommandFailedError(result)


class DeviceNotFoundError(Exception):
    """Raised when the requested device could not be captured."""
    pass


class LibXIMCCommandFailedError(Exception):
    """Raised when a libximc command fails unexpectedly."""

    DEFAULT_MSG = 'LibXIMC command failed, error code {0} ({1})'

    def __init__(self, result: int, message: str = DEFAULT_MSG):
        info = 'Unknown'
        match result:
            case pyximc.Result.Ok:
                info = 'Ok'
            case pyximc.Result.Error:
                info = 'Error'
            case pyximc.Result.NotImplemented:
                info = 'NotImplemented'
            case pyximc.Result.ValueError:
                info = 'ValueError'
            case pyximc.Result.NoDevice:
                info = 'NoDevice'
        super().__init__(message.format(result, info))


class NoUserUnitError(Exception):
    """Raised when the user tries to use user units but never defined them."""
    pass
