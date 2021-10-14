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
        self.serial = serial
        self.device_id = self._capture_device_with_serial(serial)

    @staticmethod
    def _capture_device_with_serial(serial: int) -> int:
        """

        Args:
            serial: The serial number of the device you're looking for.

        Returns: the device number (int) of the device.
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
        controller_name_c = pyximc.controller_name_t()
        serial_num_c = pyximc.c_uint()
        # Iterate over all devices, finding their friendly controller name.
        for device_index in range(0, device_count):
            # Get the device's COM port name
            port_name = libximc.get_device_name(device_enum, device_index)

            # Get the device's controllers friendly name
            result = libximc.get_enumerate_device_controller_name(
                device_enum,
                device_index,
                # Pass-by-reference to store controller name in controller_name
                pyximc.byref(controller_name_c)
            )
            if result != pyximc.Result.Ok:
                _logger.debug(f'Failed to retrieve controller name for device '
                              f'#{device_index}, error: {result}')
                continue
            friendly_name = controller_name_c.ControllerName.decode('utf-8')

            # Get the device's serial number
            result = libximc.get_enumerate_device_serial(
                device_enum,
                device_index,
                pyximc.byref(serial_num_c)
            )
            if result != pyximc.Result.Ok:
                _logger.debug(f'Failed to retrieve serial number for device '
                              f'#{device_index}, error: {result}')
                continue
            serial_num = serial_num_c.value

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
                return device_id

        # Raise an error because we couldn't find our device
        err_msg = f'Could not find device with serial {serial}'
        _logger.error(err_msg)
        raise DeviceNotFoundError(err_msg)


class DeviceNotFoundError(Exception):
    pass


def preamble():
    """Sets up pyximc to start manipulating motors."""

    # Check the version of the XIMC library
    sbuf = pyximc.create_string_buffer(64)
    libximc.ximc_version(sbuf)
    _logger.info("pyXIMC version: " + sbuf.raw.decode().rstrip("\0"))

    # If we were using networked controllers, here is where you'd set a bindy
    #  key. There's information on it in the pyximc test scripts.
