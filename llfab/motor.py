"""
Module for controlling the motors through pyximc.
"""
import warnings
from typing import TypeAlias, Optional, Tuple

import enum
import logging
import math

import pyximc
from pyximc import lib as libximc

from llfab import util

# --- Module Definitions ------------------------------------------------------
# TODO: Does this make the code more or less clear?
UnitPosition: TypeAlias = float
StepPosition: TypeAlias = Tuple[int, int]
Position: TypeAlias = UnitPosition | StepPosition


_logger = logging.getLogger(__name__)


# --- Class: MotorController --------------------------------------------------
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
        self.unit_name: Optional[str] = None
        self._free: bool = True  # If I am ._free, I don't need to be .free()d.

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
                self._free = False  # I have a device_id, I need to be freed.
                return

        # Raise an error because we couldn't find our device
        err_msg = f'Could not find device with serial {serial}'
        _logger.error(err_msg)
        raise DeviceNotFoundError(err_msg)

    def _set_user_unit(self, steps_per_unit: float, unit_name: str = ''):
        """Set the motor's user unit to scale by steps_per_unit.

        Args:
            steps_per_unit (float): The multiplier to convert units to steps
            unit_name (str): The name of the unit, helpful for debugging.
        """
        _logger.debug(f'Setting user unit of motor {self.name} to unit '
                      f'{unit_name}, {steps_per_unit} steps per unit')
        self.unit = pyximc.calibration_t()
        self.unit.A = steps_per_unit
        self.unit_name = unit_name

    # --- De-Initialization ---------------------------------------------------
    def free(self):
        """Releases the captured device."""
        if self._free:
            _logger.debug(f'No device captured, nothing to free.')
            return
        _logger.debug(f'Freeing motor {self.name}.')
        device_id_struct = pyximc.c_int(self.device_id)
        result = libximc.close_device(
            pyximc.byref(device_id_struct)
        )
        if result != pyximc.Result.Ok:
            raise LibXIMCCommandFailedError(result)
        self._free = True

    # --- Utility -------------------------------------------------------------
    def _unit_to_step(self, amount: UnitPosition) -> StepPosition:
        """Convert a position or offset from user units to steps.

        Args:
            amount: The position or offest in user units.

        Returns:
            An (int, int) tuple of (steps, microsteps) corresponding to the
            position/offset specified.

        Raises:
            NoUserUnitError: If the user has not given user units.
        """
        if not self.unit:
            raise NoUserUnitError()
        return \
            int((self.unit.A * amount) // 1), \
            int(256 * ((self.unit.A * amount) % 1))

    def _step_to_unit(self, step: StepPosition) -> UnitPosition:
        """Convert a position or offset from steps to user units.

        Args:
            step: A tuple (int, int) of (steps, microsteps).

        Returns:
            An float of the user unit representation of that position.

        Raises:
            NoUserUnitError: If the user has not given user units.
        """
        if not self.unit:
            raise NoUserUnitError()

        # Allow the user to not give a tuple, but don't suggest this behavior.
        if not isinstance(step, (tuple, list)): step = (step, 0)
        # Note that step[1], the microsteps, is an unsigned int, and it should
        #  have the same sign as step[0].
        # This will allow the user to supply floating point steps, but
        #  again, I'm not going to suggest this behavior.
        step_float = (step[0] * 256 + math.copysign(step[1], step[0])) / 256
        return step_float / self.unit.A

    def _parse_position(self, *,
                        pos: Optional[Position] = None,
                        unit: Optional[UnitPosition] = None,
                        step: Optional[StepPosition] = None,) -> StepPosition:
        """Converts user-supplied position intelligently into steps (int, int).

        - It will throw warnings if the behavior might be unexpected.
        - Exactly one of the parameters can be supplied.
        - It will handle it steps is an int or a float, but this is not
            recommended usage.

        Args:
            pos: The position, in either units or steps.
            unit: The position in user units.
            step: The position in steps.

        Returns:
            The position in (int, int) corresponding to (steps, microsteps).

        Raises:
            NoUserUnitError: If you supply unit but don't define user units.
        """
        if sum([pos is not None, unit is not None, step is not None]) != 1:
            raise TypeError('Exactly one of (pos, unit, step) should be given.')

        if pos is not None:
            if self.unit is None:
                # If no user unit is defined, assume they're giving a step.
                step = pos
            elif isinstance(pos, (list, tuple)):
                # If they give a tuple/list, it's definitely a step.
                step = pos
            else:
                # If they defined a unit and didn't give tuple/list, then they
                #  probably want to use user units.
                unit = pos

        if unit is not None:
            return self._unit_to_step(unit)
        elif step is not None:
            if isinstance(step, (tuple, list)):
                return tuple(step)
            elif isinstance(step, int):
                return step, 0
            elif isinstance(step, float):
                return int(step), int((step % 1) * 256)
            else:
                raise TypeError('Must supply int, float, or a pair thereof.')
        assert False, 'Unreachable code'

    # TODO: Rename to rail (?).
    def _bound_position(self, step: StepPosition, rail=False) -> StepPosition:
        """Check the position against the bounds, if bounds are supplied.

        Args:
            step: The (int, int) step position to check bounds for.
            rail: If the position is out of bounds, return the bound instead
                of raising an error.

        Returns:
            The position to go to in (steps, microsteps). If rail=True, then
            it might be the maximum/minimum possible. Otherwise, it will be
            exactly what you put in.

        Raises:
            PositionOutOfBoundsError: If rail=False and you supply a position
                that's out of bounds.
        """
        raise NotImplementedError()
        if self.bounds[0] is not None:
            lower = self.bounds[0][0] + self.bounds[0][1] / 256
            if step[0] + step[1] / 256 < lower:
                if rail: return self.bounds[0]
                raise PositionOutOfBoundsError(f'Position {step} below lower '
                                               f'bound {self.bounds[0]}')
        if self.bounds[1] is not None:
            upper = self.bounds[1][0] + self.bounds[1][1] / 256
            if step[0] + step[1] / 256 > upper:
                if rail: return self.bounds[1]
                raise PositionOutOfBoundsError(f'Position {step} above upper '
                                               f'bound {self.bounds[1]}')
        return step

    # --- Information (Getters) -----------------------------------------------
    def _get_boundaries(self) -> (StepPosition, StepPosition):
        """Get the boundaries of the motor.

        Returns: A tuple of tuples, representing the upper and lower bounds,
            in (step, microstep) format.
        """
        edges_settings_struct = pyximc.edges_settings_t()

        result = libximc.get_edges_settings(
            self.device_id,
            pyximc.byref(edges_settings_struct)
        )
        if result != pyximc.Result.Ok:
            _logger.debug(
                f'Failed to retrieve boundary settings for device (...), '
                f'error: {result}')

        border_flags = edges_settings_struct.BorderFlags
        ender_flags = edges_settings_struct.EnderFlags
        left_border_steps = edges_settings_struct.LeftBorder
        left_border_microsteps = edges_settings_struct.uLeftBorder
        right_border_steps = edges_settings_struct.RightBorder
        u_right_border_microsteps = edges_settings_struct.uRightBorder

        # BorderFlags has the following values:
        #  Even: It's not actually using the borders
        #  3: It's just stopping on the left border
        #  5: It's just stopping on the right border
        #  7: It's stopping on both borders.
        # If they are 8 or greater, subtract that 8 out, I don't know what it
        #  does. Look at the Border Flags section of the manual for more info.
        if border_flags != 7:
            _logger.warning(f'Motor {self.name} is not stopping at both '
                            f'borders, it has border flag {border_flags}.')

        return(
            (left_border_steps, left_border_microsteps),
            (right_border_steps, u_right_border_microsteps)
        )

    def _get_position(self) -> StepPosition:
        """Get the motor's current position in steps.

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

    # TODO: Let this also give it in steps?
    def get_boundaries(self) -> (UnitPosition, UnitPosition):
        """Get the motor's boundaries, in user units."""
        lower_boundary_steps, upper_boundary_steps = self._get_boundaries()
        return (
            self._step_to_unit(lower_boundary_steps),
            self._step_to_unit(upper_boundary_steps),
        )

    # TODO: Let this also give it in steps?
    def get_position(self) -> UnitPosition:
        """Get the motor's current position in user units.

        Returns:
            position (float): The position in user units.
        """
        return self._step_to_unit(self._get_position())

    # @property
    # def bounds(self) -> (UnitPosition, UnitPosition):
    #     return self.get_position()
    #
    # @property
    # def position(self) -> UnitPosition:
    #     return self.get_position()

    # --- Settings Customization (Setters) ------------------------------------
    def set_zero(self):
        """Set the current position as zero."""
        _logger.debug(f'Defining current position of {self.name} as zero')
        result = libximc.command_zero(
            self.device_id,
        )
        if result != pyximc.Result.Ok:
            raise LibXIMCCommandFailedError(result)

    # --- Movement Functions --------------------------------------------------
    # TODO: Should I have separate move functions for user units and steps,
    #  or should it be a flag?
    def move_to(self, pos: Optional[Position] = None, *,
                unit: Optional[UnitPosition] = None,
                step: Optional[StepPosition] = None,
                rail: bool = False):
        """Move to the specified location, in user units (absolute position).

        Exactly one of pos, unit, and step should be supplied.

        Args:
            pos: The position to move to, in user units or steps.
            unit: The position to move to, in user units.
            step: The position to move to, in (step, microstep) format.
            rail: If True, then instead of throwing an error if you go out of
                bounds, it will move as far as allowed.

        Raises:
            LibXIMCCommandFailedError: If the movement fails
            NoUserUnitError: If you supply unit but don't define user units.
            PositionOutOfBoundsError: If rail=False and you supply a position
                that's out of bounds.
        """
        step, microstep = self._parse_position(pos=pos,
                                               unit=unit,
                                               step=step,)
        # step, microstep = self._bound_position(step=(step, microstep),
        #                                        rail=rail,)

        if self.unit is not None:
            _logger.debug(
                f'Moving {self.name} to step {step} + {microstep}/256 '
                f'({self._step_to_unit((step, microstep))} {self.unit_name})'
            )
        else:
            _logger.debug(
                f'Moving {self.name} to step {step} + {microstep}/256'
            )

        result = libximc.command_move(
            self.device_id,
            step,
            microstep,
        )
        if result != pyximc.Result.Ok:
            raise LibXIMCCommandFailedError(result)

    def move_by(self, by: Optional[Position] = None, *,
                unit: Optional[UnitPosition] = None,
                step: Optional[StepPosition] = None,
                rail: bool = False):
        """Move by the specified number of user units (relative positioning).

        Args:
            by (float): The number of user units to move by.
            unit (float): The amount to move by, in user units.
            step (float): The amount to move by, in (step, microstep) format.
            rail: If True, then instead of throwing an error if you go out of
                bounds, it will move as far as allowed.

        Raises:
            LibXIMCCommandFailedError: If the movement fails
            NoUserUnitError: If the user never defined user units.
        """
        step, microstep = self._parse_position(pos=by,
                                               unit=unit,
                                               step=step,)
        # to_step, to_microstep = self.get_position_step()
        # to_step += step
        # to_microstep += microstep
        # step, microstep = self._bound_position(step=(to_step, to_microstep),
        #                                        rail=False,)
        # TODO: How should this behave? Should it rail? Should I use movr or
        #  move? Should I try to get calb_move working? It gave me -3,
        #  ValueError, no matter what I put into it. :L

        if self.unit is not None:
            _logger.debug(
                f'Moving {self.name} by {step} + {microstep}/256 steps'
                f'({self._step_to_unit((step, microstep))} {self.unit_name})'
            )
        else:
            _logger.debug(
                f'Moving {self.name} by {step} + {microstep}/256 steps'
            )

        result = libximc.command_movr(
            self.device_id,
            step,
            microstep,
        )
        if result != pyximc.Result.Ok:
            raise LibXIMCCommandFailedError(result)

    @util.depreciate
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

    @util.depreciate
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

    # --- Magic Methods -------------------------------------------------------
    def __del__(self):
        self.free()


# --- Errors ------------------------------------------------------------------
class DeviceNotFoundError(Exception):
    """Raised when the requested device could not be captured."""
    pass


class LibXIMCCommandFailedError(Exception):
    """Raised when a libximc command fails unexpectedly."""

    DEFAULT_MSG = 'LibXIMC command failed, error code {0} ({1}).'

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


class PositionOutOfBoundsError(Exception):
    """Raised when the user tries to move to a position out of bounds."""
    pass
