from ctypes import byref, c_float

from pyximc import get_position_t, Result, get_position_calb_t, status_t, \
    MvcmdStatus
import scripts.test_extended.info_io
from scripts.test_extended.config import user_unit


def test_get_position(lib, device_id, mode=1):
    """
    Obtaining information about the position of the positioner.

    This function allows you to get information about the current positioner coordinates,
    both in steps and in encoder counts, if it is set.
    Also, depending on the state of the mode parameter, information can be obtained in user units.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
        mode: mode in feedback counts or in user units. (Default value = 1)
    """

    # print("\nRead position")
    if mode:
        x_pos = get_position_t()
        result = lib.get_position(device_id, byref(x_pos))
        if result == Result.Ok:
            print("Position: {0} steps, {1} microsteps".format(x_pos.Position, x_pos.uPosition), end="\r")
        return x_pos.Position, x_pos.uPosition
    else:
        x_pos = get_position_calb_t()
        result = lib.get_position_calb(device_id, byref(x_pos), byref(user_unit))
        if result == Result.Ok:
            print("Position: {0} user unit".format(x_pos.Position), end="\r")
        return x_pos.Position, 0


def test_left(lib, device_id):
    """
    Move to the left.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    """

    print("\nMoving left")
    result = lib.command_left(device_id)


def test_right(lib, device_id):
    """
    Move to the right.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    """

    print("\nMoving right")
    result = lib.command_right(device_id)


def test_move(lib, device_id, distance, udistance, mode=1):
    """
    Move to the specified coordinate.

    Depending on the mode parameter, you can set coordinates in steps or feedback counts, or in custom units.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
        distance: the position of the destination.
        udistance: destination position in micro steps if this mode is used.
        mode: mode in feedback counts or in user units. (Default value = 1)
    """

    if mode:
        print("\nGoing to {0} steps, {1} microsteps".format(distance, udistance))
        result = lib.command_move(device_id, distance, udistance)
    else:
        # udistance is not used for setting movement in custom units.
        print("\nMove to the position {0} specified in user units.".format(distance))
        result = lib.command_move_calb(device_id, c_float(distance), byref(user_unit))


def test_movr(lib, device_id, distance, udistance, mode=1):
    """
    The shift by the specified offset coordinates.

    Depending on the mode parameter, you can set coordinates in steps or feedback counts, or in custom units.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
        distance: size of the offset in steps.
        udistance: Size of the offset in micro steps.
        mode: Default value = 1)
    """

    if mode:
        print("\nShift to {0} steps, {1} microsteps".format(distance, udistance))
        result = lib.command_movr(device_id, distance, udistance)
    else:
        # udistance is not used for setting movement in custom units.
        print("\nShift to the position {0} specified in user units.".format(distance))
        result = lib.command_movr_calb(device_id, c_float(distance), byref(user_unit))


def test_wait_for_stop(lib, device_id, interval):
    """
    Waiting for the movement to complete.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
        interval: step of the check time in milliseconds.
    """

    print("\nWaiting for stop")
    result = lib.command_wait_for_stop(device_id, interval)
    print("Result: " + repr(result))


def flex_wait_for_stop(lib, device_id, msec, mode=1):
    """
    This function performs dynamic output coordinate in the process of moving.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
        msec: Pause between reading the coordinates.
        mode: data mode in feedback counts or in user units. (Default value = 1)
    """

    stat = status_t()
    stat.MvCmdSts |= 0x80
    while (stat.MvCmdSts & MvcmdStatus.MVCMD_RUNNING > 0):
        result = scripts.test_extended.info_io.get_status(device_id, byref(stat))
        if result == Result.Ok:
            test_get_position(lib, device_id, mode)
            lib.msec_sleep(msec)