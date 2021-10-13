from ctypes import byref, string_at, c_uint

from pyximc import device_information_t, Result, status_t, stage_information_t, \
    motor_settings_t, StateFlags, MotorTypeFlags, sync_in_settings_t
from scripts.test_extended.config import user_unit


def test_info(lib, device_id):
    """
    Reading information about the device.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    """

    print("\nGet device info")
    x_device_information = device_information_t()
    result = lib.get_device_information(device_id, byref(x_device_information))
    print("Result: " + repr(result))
    if result == Result.Ok:
        print("Device information:")
        print(" Manufacturer: " +
              repr(string_at(x_device_information.Manufacturer).decode()))
        # print(" ManufacturerId: " +
        #        repr(string_at(x_device_information.ManufacturerId).decode()))
        # print(" ProductDescription: " +
        #        repr(string_at(x_device_information.ProductDescription).decode()))
        print(" Hardware version: " + repr(x_device_information.Major) + "." + repr(x_device_information.Minor) +
              "." + repr(x_device_information.Release))
        # print(" Major: " + repr(x_device_information.Major))
        # print(" Minor: " + repr(x_device_information.Minor))
        # print(" Release: " + repr(x_device_information.Release))


def test_status(lib, device_id):
    """
    A function of reading status information from the device

    You can use this function to get basic information about the device status.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    """

    print("\nGet status")
    x_status = status_t()
    result = lib.get_status(device_id, byref(x_status))
    print("Result: " + repr(result))
    if result == Result.Ok:
        print("Status.Ipwr: " + repr(x_status.Ipwr))
        print("Status.Upwr: " + repr(x_status.Upwr))
        print("Status.Iusb: " + repr(x_status.Iusb))
        print("Status.Flags: " + repr(hex(x_status.Flags)))


def get_status(lib, device_id):
    """
    A function of reading status information from the device

    You can use this function to get basic information about the device status.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    """

    x_status = status_t()
    result = lib.get_status(device_id, byref(x_status))
    if result == Result.Ok:
        return x_status
    else:
        return None


def get_stage_information(lib, device_id):
    """
    Read information from the EEPROM of the progress bar if it is installed.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    """

    x_stage_inf = stage_information_t()
    result = lib.get_stage_information(device_id, byref(x_stage_inf))
    if result == Result.Ok:
        return x_stage_inf
    else:
        return None


def get_motor_settings(lib, device_id):
    """
    Receiving the configuration of the motor.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    """

    x_motor_settings = motor_settings_t()
    result = lib.get_motor_settings(device_id, byref(x_motor_settings))
    if result == Result.Ok:
        return x_motor_settings
    else:
        return None


def test_serial(lib, device_id):
    """
    Reading the device's serial number.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    """

    # print("\nReading serial")
    x_serial = c_uint()
    result = lib.get_serial_number(device_id, byref(x_serial))
    if result == Result.Ok:
        print(" Serial: " + repr(x_serial.value))


def test_get_move_settings(lib, device_id, mvst, mode=1):
    """
    Read the move settings.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
        mvst: the structure with parameters of movement.
        mode: data mode in feedback counts or in user units. (Default value = 1)
    """

    # Get current move settings from controller
    if mode:
        result = lib.get_move_settings(device_id, byref(mvst))
    else:
        result = lib.get_move_settings_calb(device_id, byref(mvst), byref(user_unit))
    # Print command return status. It will be 0 if all is OK
    if result == Result.Ok:
        print("Current speed: " + repr(mvst.Speed))
        print("Current acceleration: " + repr(mvst.Accel))
        print("Current deceleration: " + repr(mvst.Decel) + "\n")


def test_set_move_settings(lib, device_id, mvst, mode=1):
    """
    Write the move settings.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
        mvst: the structure with parameters of movement.
        mode: data mode in feedback counts or in user units. (Default value = 1)
    """

    # Get current move settings from controller
    if mode:
        result = lib.set_move_settings(device_id, byref(mvst))
    else:
        result = lib.set_move_settings_calb(device_id, byref(mvst), byref(user_unit))


def test_eeprom(lib, device_id):
    """
    Checks for the presence of EEPROM. If it is present, it displays information.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    """

    print("Test EEPROM")
    status = get_status(lib, device_id)
    if status != None:
        if int(repr(status.Flags)) and StateFlags.STATE_EEPROM_CONNECTED:
            print("EEPROM CONNECTED")
            stage_information = get_stage_information(lib, device_id)
            print("PartNumber: " + repr(string_at(stage_information.PartNumber).decode()))
            motor_settings = get_motor_settings(lib, device_id)
            if int(repr(motor_settings.MotorType)) == MotorTypeFlags.MOTOR_TYPE_STEP:
                print("Motor Type: STEP")
            elif int(repr(motor_settings.MotorType)) == MotorTypeFlags.MOTOR_TYPE_DC:
                print("Motor Type: DC")
            elif int(repr(motor_settings.MotorType)) == MotorTypeFlags.MOTOR_TYPE_BLDC:
                print("Motor Type: BLDC")
            else:
                print("Motor Type: UNKNOWN")
        else:
            print("EEPROM NO CONNECTED")


def test_sync_settings(lib, device_id):
    """


    Args:
        lib: param device_id:
        device_id:
    """
    sync_settings = sync_in_settings_t()
    result = lib.get_sync_in_settings(device_id, byref(sync_settings))
    sync_settings.Position = 500
    sync_settings.Speed = 500
    result = lib.set_sync_in_settings(device_id, byref(sync_settings))


def motor_settings(lib, device_id):
    """


    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    """

    get_motor_settings(device_id, motor_settings_t * motor_settings)