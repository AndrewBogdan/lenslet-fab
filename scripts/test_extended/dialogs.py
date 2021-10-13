import os
import platform
from ctypes import byref

import keyboard
import netifaces

from pyximc import lib, ximc_dir, EnumerateFlags, controller_name_t, \
    extio_settings_t, ExtioModeFlags, ExtioSetupFlags, feedback_settings_t, \
    Result, FeedbackType, FeedbackFlags, edges_settings_t, \
    BorderFlags, EnderFlags, move_settings_t, move_settings_calb_t
from scripts.test_extended.config import user_name, key_esc, eng, user_unit
from scripts.test_extended.info_io import test_get_move_settings, \
    test_set_move_settings, test_eeprom
from scripts.test_extended.move import test_get_position, test_left, test_right, \
    test_move, test_movr, test_wait_for_stop, flex_wait_for_stop


def getch_new():
    """Processing keystrokes by scancode. Provides work independent of the national layout."""

    scan_key = {2: 49, 3: 50, 4: 51, 5: 52, 6: 53, 7: 54, 8: 55, 9: 56, 10: 57,
                11: 48, 35: 72,
                19: 82, 44: 90, 32: 68, 31: 83, 21: 89, 16: 81, 50: 77, 46: 67,
                33: 70, 18: 69, 22: 85,
                38: 76, 23: 73, 24: 79}

    if 1:
        s1 = keyboard.read_key(False)
        s1 = keyboard.read_key(False)
        a = keyboard.key_to_scan_codes(s1)[0]
        try:
            s = chr(scan_key[a])
        except:
            if len(s1) == 1:
                s1 = "";
            s = " "
            print("Invalid key {}".format(s1))

        keyboard.send(key_esc, do_press=True, do_release=True)
        return s


def getch():
    """Select the type of keyboard processing depending on your rights."""

    if user_name != "root":
        raise NotImplementedError #return getch1()
    else:
        return getch_new()


def device_selection_dialog():
    """
    Device selection Manager.

    Set bindy (network) keyfile. Must be called before any call to "enumerate_devices" or "open_device" if you
    wish to use network-attached controllers. Accepts both absolute and relative paths, relative paths are resolved
    relative to the process working directory. If you do not need network devices then "set_bindy_key" is optional.
    In Python make sure to pass byte-array object to this function (b"string literal").
    Follow the on-screen instructions to change the settings.
    """

    print("What XIMC device do you plan to open?")
    print("Press the key:")
    print("1 - is COM;")
    print("2 - virtual controller;")
    print("3 - network controller;")
    print("4 - search for all available devices.")
    print("What device do you plan to open?")
    key_press = getch()
    port_name = None
    head_port = None
    if ord(key_press) == 49:  # """ Press "1" COM """
        if platform.system() == "Windows":
            print("Enter the port number: COM")
            head_port = "xi-com:\\\.\COM"
        else:
            print("Enter the port number: dev/tty.s")
            head_port = "xi-com:/dev/tty.s"
        port_name = input_new()
    elif ord(key_press) == 50:  # """ Press "2" virtual controller """
        print("Enter the name of the virtual device:")
        head_port = "xi-emu:///"
        port_name = input_new()
    elif ord(key_press) == 51:  # """ Press "3" network controller """
        print("Enter the device's network address:")
        print("Example:192.168.0.1/89ABCDEF")
        lib.set_bindy_key(
            os.path.join(ximc_dir, "win32", "keyfile.sqlite").encode("utf-8"))
        head_port = "xi-net://"
        port_name = input_new()
    elif ord(key_press) == 52:  # Press "4" search for all available devices
        # Set bindy (network) keyfile. Must be called before any call to "enumerate_devices" or "open_device" if you
        # wish to use network-attached controllers. Accepts both absolute and relative paths, relative paths are resolved
        # relative to the process working directory. If you do not need network devices then "set_bindy_key" is optional.
        # In Python make sure to pass byte-array object to this function (b"string literal").
        print("Wait for the search to complete...")
        res = lib.set_bindy_key(
            os.path.join(ximc_dir, "win32", "keyfile.sqlite").encode("utf-8"))
        if res < 0:
            res = lib.set_bindy_key("keyfile.sqlite".encode("utf-8"))
            if res < 0:
                print(
                    "The keyfile.sqlite file was not found. It is located in the ximc\win32 folder. Copy it to the current folder.")
                exit(1)
        # This is device search and enumeration with probing. It gives more information about devices.
        probe_flags = EnumerateFlags.ENUMERATE_PROBE + EnumerateFlags.ENUMERATE_NETWORK  # + EnumerateFlags.ENUMERATE_ALL_COM
        print("Search on network interfaces")
        interfaces = netifaces.interfaces()
        device_set = set()
        for interface in interfaces:
            addrs = netifaces.ifaddresses(interface).get(netifaces.AF_INET)
            if addrs is None:
                continue
            enum_hints = "addr=\nadapter_addr=" + addrs[0]["addr"]

            print(addrs[0]["addr"])

            if type(enum_hints) is str:
                enum_hints = enum_hints.encode()

            devenum = lib.enumerate_devices(probe_flags, enum_hints)

            dev_count = lib.get_device_count(devenum)
            controller_name = controller_name_t()
            for dev_ind in range(0, dev_count):
                enum_name = lib.get_device_name(devenum, dev_ind)
                device_set.add(enum_name)

        device_list = list(device_set)
        for i in range(0, len(device_list)):
            print("Enumerated device #{} name (port name){}: ".format(i,
                                                                      device_list[
                                                                          i]))
        print("Select the device number:#")
        key_press = getch()
        try:
            if int(key_press) < len(device_list):
                head_port = device_list[int(key_press)]
                port_name = b""
            else:
                print("A device with this number:#", int(key_press),
                      " is not in the list.")
                head_port = ""
                port_name = ""
        except:
            print("Input error")
            head_port = ""
            port_name = ""
    else:
        head_port = ""
        port_name = ""
    dev_port = head_port + port_name
    return dev_port


def device_movement_actions_dialog(lib, device_id, mode=1):
    """
    The Manager motion control.

    Allows you to move both in feedback counts and in user units.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
        mode: data mode in feedback counts or in user units. (Default value = 1)
    Follow the on-screen instructions to change the settings.
    """

    key_press = "1"
    # current_speed = test_get_speed(lib, device_id)
    position, uposition = test_get_position(lib, device_id, mode)
    print("\n")
    while (ord(key_press) != 81 and ord(key_press) != 113):  # Press "q" - quit
        # if 1:
        print("Actions with the XIMC device")
        print("Q or q keys - return to the main menu;")
        print("4 - move to the left. Press and hold the key;")
        print("6 - move to the right. Press and hold the key;")
        print("M or m key - move to position(mov);")
        print("R or r keys - position shift(movr);")
        print("H or h keys - HOME position;")
        print("Z or z keys - ZERO position;")
        print("S or s keys - move setting;")
        print("Selected action.")
        key_press = getch()
        if ord(key_press) == 77 or ord(key_press) == 109:  # Press "M"
            print("Enter a position:")
            if mode:
                position = int(input_new())
            else:
                position = float(input_new())
            test_move(lib, device_id, position, 0, mode)
            # Interactive output of coordinates.
            flex_wait_for_stop(lib, device_id, 10, mode)
        elif ord(key_press) == 82 or ord(key_press) == 114:  # Press "R"
            print("Enter a shift position:")
            if mode:
                sposition = int(input_new())
            else:
                sposition = float(input_new())
            test_movr(lib, device_id, sposition, 0, mode)
            # Interactive output of coordinates.
            flex_wait_for_stop(lib, device_id, 10, mode)
        elif ord(key_press) == 52:  # Press "4" move to the left
            print("Move to the left:")
            test_left(lib, device_id)
            print("To stop, press any key.")
            while ord(key_press) == 52:
                key_press = getch()
            lib.command_sstp(device_id)
        elif ord(key_press) == 54:  # Press "6" move to the left
            print("Move to the left:")
            test_right(lib, device_id)
            print("To stop, press any key.")
            while ord(key_press) == 54:
                key_press = getch()
            lib.command_sstp(device_id)
        elif ord(key_press) == 72 or ord(key_press) == 104:  # Press "H"
            print("HOME position:")
            lib.command_home(device_id)
            flex_wait_for_stop(lib, device_id, 10, mode)
        elif ord(key_press) == 90 or ord(key_press) == 122:  # Press "Z"
            print("Zero position:")
            lib.command_zero(device_id)
        elif ord(key_press) == 83 or ord(key_press) == 115:  # Press "S"
            test_move_settings(lib, device_id, mode)
        print("Wait for the movement to finish...")
        test_wait_for_stop(lib, device_id, 10)
        position, uposition = test_get_position(lib, device_id, mode)
        print("\n")


def device_actions_dialog(lib, device_id):
    """
    The main Manager of the example.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    Follow the on-screen instructions to change the settings.
    """

    key_press = "1"
    print(" ")
    while (ord(key_press) != 81 and ord(key_press) != 113):  # Press "q" - quit
        print("Actions with the XIMC device")
        print("Press the case sensitive key :")
        print("Q or q keys - exit;")
        print("M or m key - movement;")
        print("C or c key - calb movement;")
        print("I or i keys - external I/O EXTIO;")
        print("E or e keys - EEPROM;")
        print("S or s keys - setting;")
        print("Selected action.")
        print(" ")
        key_press = getch()
        # print(str(key_press), ord(key_press))
        if ord(key_press) == 77 or ord(key_press) == 109:  # Press "M" movement
            device_movement_actions_dialog(lib, device_id)
        if ord(key_press) == 67 or ord(key_press) == 99:  # Press "C" movement
            device_movement_actions_dialog(lib, device_id, 0)
        elif ord(key_press) == 73 or ord(
                key_press) == 105:  # Press "I" external I/O EXTIO
            test_extio(lib, device_id)
        elif ord(key_press) == 69 or ord(key_press) == 101:  # Press "E" EEPROM
            test_eeprom(lib, device_id)
        elif ord(key_press) == 83 or ord(key_press) == 115:  # Press "S"
            gl_settings(lib, device_id)
        print(" ")


def test_extio(lib, device_id):
    """
    External input / output settings Manager.

    Follow the on-screen instructions to change the settings.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    """

    print("Use output as input or output?")
    print("Press the case sensitive key:")
    print("I or i keys - input;")
    print("O or o keys - output;")
    print("R or r keys - inverted output;")
    key_press = getch()
    extio_settings = extio_settings_t()
    result = lib.get_extio_settings(device_id, byref(extio_settings))
    if ord(key_press) == 73 or ord(key_press) == 105:  # Press "I" input
        extio_settings.EXTIOSetupFlags = 0
        print("The input destination")
        print("Press the key:")
        print("1 - EXTIO_SETUP_MODE_IN_STOP")
        print("2 - EXTIO_SETUP_MODE_IN_PWOF;")
        print("3 - EXTIO_SETUP_MODE_IN_MOVR;")
        print("4 - EXTIO_SETUP_MODE_IN_HOME;")
        print("5 - EXTIO_SETUP_MODE_IN_ALARM;")
        key_press = getch()
        print(" ")
        print(
            "When the logical level is applied to the input, the following will be performed:")
        if ord(key_press) == 49:  # Press "1" EXTIO_SETUP_MODE_IN_STOP
            extio_settings.EXTIOModeFlags = ExtioModeFlags.EXTIO_SETUP_MODE_IN_STOP
            print("Stop moving")
        elif ord(key_press) == 50:  # Press "2" EXTIO_SETUP_MODE_IN_PWOF
            extio_settings.EXTIOModeFlags = ExtioModeFlags.EXTIO_SETUP_MODE_IN_PWOF
            print("Power OFF")
        elif ord(key_press) == 51:  # Press "3" EXTIO_SETUP_MODE_IN_MOVR
            extio_settings.EXTIOModeFlags = ExtioModeFlags.EXTIO_SETUP_MODE_IN_MOVR
            print("Move with MOVR")
        elif ord(key_press) == 52:  # Press "4" EXTIO_SETUP_MODE_IN_HOME
            extio_settings.EXTIOModeFlags = ExtioModeFlags.EXTIO_SETUP_MODE_IN_HOME
            print("HOME")
        elif ord(key_press) == 53:  # Press "5" EXTIO_SETUP_MODE_IN_ALARM
            extio_settings.EXTIOModeFlags = ExtioModeFlags.EXTIO_SETUP_MODE_IN_ALARM
            print("Alarm state")
    elif ord(key_press) == 79 or ord(key_press) == 82 or ord(
            key_press) == 111 or ord(
            key_press) == 114:  # Press "O" or "R" output
        extio_settings.EXTIOSetupFlags = ExtioSetupFlags.EXTIO_SETUP_OUTPUT
        if ord(key_press) == 82:
            extio_settings.EXTIOSetupFlags &= ExtioSetupFlags.EXTIO_SETUP_INVERT
        print("The output destination")
        print("Press the key:")
        print("2 - EXTIO_SETUP_MODE_OUT_MOVING;")
        print("3 - EXTIO_SETUP_MODE_OUT_ALARM;")
        print("4 - EXTIO_SETUP_MODE_OUT_MOTOR_ON;")
        key_press = getch()
        print(" ")
        print("In the case:")
        if ord(key_press) == 50:  # Press "2" EXTIO_SETUP_MODE_OUT_MOVING
            extio_settings.EXTIOModeFlags = ExtioModeFlags.EXTIO_SETUP_MODE_OUT_MOVING
            print("the beginning of the movement")
        elif ord(key_press) == 51:  # Press "3" EXTIO_SETUP_MODE_OUT_ALARM
            extio_settings.EXTIOModeFlags = ExtioModeFlags.EXTIO_SETUP_MODE_OUT_ALARM
            print("alarm state")
        elif ord(key_press) == 52:  # Press "4" EXTIO_SETUP_MODE_OUT_MOTOR_ON
            extio_settings.EXTIOModeFlags = ExtioModeFlags.EXTIO_SETUP_MODE_OUT_MOTOR_ON
            print("power supply to the motor")
        print("the output state will change.")
    result = lib.set_extio_settings(device_id, byref(extio_settings))


def gl_settings(lib, device_id):
    """
    Manager of the controller settings.

    This function, among other settings, allows you to load the coordinate correction table.
    Follow the on-screen instructions to change the settings.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    note:
    The device_id parameter in this function is a C pointer, unlike most library functions that use this parameter
    """

    key_press = "1"
    while (ord(key_press) != 81 and ord(key_press) != 113):  # Press "q" - quit
        print("Select a group of settings:")
        print("Q or q keys - return to the main menu;")
        print("M or m keys - motion settings")
        print("C or c keys - calb motion settings")
        print("F or f keys - feedback settings;")
        print("E or e keys - edges settings")
        print("S or s keys - micro step mode settings")
        print("U or u keys - user unit settings")
        print("L or l keys - load correction table")
        print("Selected settings.")
        key_press = getch()
        if ord(key_press) == 77 or ord(key_press) == 109:  # Press "M"
            test_move_settings(lib, device_id)
        if ord(key_press) == 67 or ord(key_press) == 99:  # Press "C" movement
            test_move_settings(lib, device_id, 0)
        if ord(key_press) == 70 or ord(key_press) == 102:  # Press "F"
            print("\nFeedback settings")
            test_feedback_settings(lib, device_id)
        if ord(key_press) == 69 or ord(key_press) == 101:  # Press "E"
            print("\nEdges settings")
            test_edges_settings(lib, device_id)
        if ord(key_press) == 83 or ord(key_press) == 115:  # Press "S"
            test_microstep_mode(lib, device_id)
        if ord(key_press) == 85 or ord(key_press) == 117:  # Press "U"
            test_user_unit_mode(lib, device_id)
        print(" ")
        if ord(key_press) == 76 or ord(key_press) == 108:  # Press "L"
            print("\nLoad correction table")
            print("You can use a short or full file name.")
            namefile = input_new("Enter the file name: ")
            if type(namefile) is str:
                namefile = namefile.encode("utf-8")

            # The device_t device parameter in this function is a C pointer, unlike most library functions that use this parameter
            # result = lib.load_correction_table(byref(cast(device_id, POINTER(c_int))), namefile)  # This function is deprecated. Use the function
            result = lib.set_correction_table(device_id, namefile)
            if result < 0:
                print("The table is not loaded, If the table was loaded, it is reset.")
            else:
                print("Table loaded successfully,")


def test_feedback_settings(lib, device_id):
    """
    View and change the feedback mode.

    To manage feedback, follow the prompts on the screen.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    """

    # Get current feedback settings from controller
    fvst = feedback_settings_t()
    result = lib.get_feedback_settings(device_id, byref(fvst))
    # Print command return status. It will be 0 if all is OK
    if result == Result.Ok:
        # Print The current feedback type
        if fvst.FeedbackType == FeedbackType.FEEDBACK_ENCODER:
            print("The current feedback type ENCODER")
        elif fvst.FeedbackType == FeedbackType.FEEDBACK_EMF:
            print("The current feedback type EMF")
        elif fvst.FeedbackType == FeedbackType.FEEDBACK_NONE:
            print("The current feedback type NONE")
        elif fvst.FeedbackType == FeedbackType.FEEDBACK_ENCODER_MEDIATED:
            print("The current feedback type ENCODER_MEDIATED")

        # Print the current settings of the encoder
        if (fvst.FeedbackFlags & FeedbackFlags.FEEDBACK_ENC_REVERSE) == FeedbackFlags.FEEDBACK_ENC_REVERSE:
            print("ENC_REVERSE")
        else:
            print("ENC_NO_REVERSE")
        if (
                fvst.FeedbackFlags & FeedbackFlags.FEEDBACK_ENC_TYPE_SINGLE_ENDED) == FeedbackFlags.FEEDBACK_ENC_TYPE_SINGLE_ENDED:
            print("FEEDBACK_ENC_TYPE_SINGLE_ENDED")
        if (
                fvst.FeedbackFlags & FeedbackFlags.FEEDBACK_ENC_TYPE_DIFFERENTIAL) == FeedbackFlags.FEEDBACK_ENC_TYPE_DIFFERENTIAL:
            print("FEEDBACK_ENC_TYPE_DIFFERENTIAL")

        # Select a new feedback mode
        print("Select a new feedback mode")
        print("5 - NONE")
        print("4 - EMF")
        print("1 - ENCODER")
        print("6 - ENCODER_MEDIATED")
        print("Any other key - cancel")
        key_press = getch()
        if ord(key_press) == 49:  # Press "1"
            print("You need to convert the movement parameters to rpm")
            print("New feedback mode  ENCODER")
            fvst.FeedbackType = FeedbackType.FEEDBACK_ENCODER
        elif ord(key_press) == 52:  # Press "4"
            print("You need to convert the movement parameters to step/s")
            print("New feedback mode  EMF")
            fvst.FeedbackType = FeedbackType.FEEDBACK_EMF
        elif ord(key_press) == 53:  # Press "5"
            print("You need to convert the movement parameters to step/s")
            print("New feedback mode  NONE")
            fvst.FeedbackType = FeedbackType.FEEDBACK_NONE
        elif ord(key_press) == 54:  # Press "6"
            print("CYou need to convert the movement parameters to rpm")
            print("New feedback mode  ENCODER_MEDIATED")
            fvst.FeedbackType = FeedbackType.FEEDBACK_ENCODER_MEDIATED

        # Invert the reverse
        print("\nR or r key invert the reverse")
        print("Any other key - cancel")
        key_press = getch()
        if ord(key_press) == 82 or ord(key_press) == 114:  # Press "R"
            if fvst.FeedbackFlags & FeedbackFlags.FEEDBACK_ENC_REVERSE == FeedbackFlags.FEEDBACK_ENC_REVERSE:
                print("ENC_NO_REVERSE")
                fvst.FeedbackFlags = fvst.FeedbackFlags & ~FeedbackFlags.FEEDBACK_ENC_REVERSE
            else:
                print("ENC_REVERSE")
                fvst.FeedbackFlags = fvst.FeedbackFlags | FeedbackFlags.FEEDBACK_ENC_REVERSE

        # Select new feedback type
        print("\nSelect a new feedback type")
        print("S or s key - SINGLE_ENDED")
        print("D or d key - DIFFERENTIAL")
        print("Any other key - cancel")
        key_press = getch()
        if ord(key_press) == 68 or ord(key_press) == 100:  # Press "D"
            fvst.FeedbackFlags = (fvst.FeedbackFlags & 0x0F) | FeedbackFlags.FEEDBACK_ENC_TYPE_DIFFERENTIAL
            print("New feedback type DIFFERENTIAL")
        elif ord(key_press) == 83 or ord(key_press) == 115:  # Press "S"
            fvst.FeedbackFlags = (fvst.FeedbackFlags & 0x0F) | FeedbackFlags.FEEDBACK_ENC_TYPE_SINGLE_ENDED
            print("New feedback type SINGLE_ENDED")
        result = lib.set_feedback_settings(device_id, byref(fvst))
        # Print command return status. It will be 0 if all is OK
        if result == Result.Ok:
            print("Error save feedback settings" + "\n")
    else:
        print("Error reading feedback settings" + "\n")


def test_microstep_mode(lib, device_id):
    """
    Setting the microstep mode. Works only for stepper motors

    Follow the on-screen instructions to change the settings.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    """

    print("\nMicrostep mode settings.")
    print("This setting is only available for stepper motors.")
    # Get current engine settings from controller
    result = lib.get_engine_settings(device_id, byref(eng))
    if result == Result.Ok:
        # Current MICROSTEP_MODE
        Microstep_Mode = ["", "MICROSTEP_MODE_FULL", "MICROSTEP_MODE_FRAC_2", "MICROSTEP_MODE_FRAC_4",
                          "MICROSTEP_MODE_FRAC_8", "MICROSTEP_MODE_FRAC_16", "MICROSTEP_MODE_FRAC_32",
                          "MICROSTEP_MODE_FRAC_64", "MICROSTEP_MODE_FRAC_128", "MICROSTEP_MODE_FRAC_256"]
        print("The mode is set to", Microstep_Mode[eng.MicrostepMode], "\n")
        # Change MicrostepMode parameter
        # (use MICROSTEP_MODE_FULL to MICROSTEP_MODE_FRAC_256 - 9 microstep modes)
        for range_val in range(len(Microstep_Mode)):
            if range_val > 0:
                print("Set mode {0} - press {1}".format(Microstep_Mode[range_val], range_val))
        try:
            in_val = int(getch())
            if in_val > 0 and in_val <= 9:
                eng.MicrostepMode = in_val
                user_unit.MicrostepMode = in_val
            print("The mode is set to", Microstep_Mode[eng.MicrostepMode])
        except:
            print("The mode is set to", Microstep_Mode[eng.MicrostepMode])
        result = lib.set_engine_settings(device_id, byref(eng))
        if result != Result.Ok:
            print("Error recording microstep mode.")
        print("")


def input_flags(flags, names_flags):
    """
    Function for entering flag values.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
        flags: param names_flags:
        names_flags:
    """

    index = -1
    for val in names_flags:
        if index >= 0:
            print(val + ": ", end="\r")

            try:
                in_val = int(getch())
            except:
                in_val = -1

            if in_val == 0 or in_val == 1:
                print(val + ": ", in_val)
                flags &= ~(1 << index)
                flags |= in_val << index
            else:
                print(val + ": ", (flags >> index) & 1)
        else:
            print(val)
        index += 1
    return flags


def test_edges_settings(lib, device_id):
    """
    View and configure the limit switch mode.

    Follow the on-screen instructions to change the settings.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    """

    # Get current feedback settings from controller
    edgst = edges_settings_t()
    result = lib.get_edges_settings(device_id, byref(edgst))
    # Print command return status. It will be 0 if all is OK
    if result == Result.Ok:
        # Print The current edge settings
        # BorderFlags
        Border_Flags = ["BorderFlags", "BORDER_IS_ENCODER", "BORDER_STOP_LEFT", "BORDER_STOP_RIGHT",
                        "BORDERS_SWAP_MISSET_DETECTION"]
        print("BorderFlags {0:x}".format(edgst.BorderFlags))
        if (edgst.BorderFlags & BorderFlags.BORDER_IS_ENCODER):
            print("BORDER_IS_ENCODER 1:The borders are set with coordinates.")
        else:
            print("BORDER_IS_ENCODER 0:The position of the borders is set by limit switches.")

        if (edgst.BorderFlags & BorderFlags.BORDER_STOP_LEFT):
            print("BORDER_STOP_LEFT 1:The motor stops when it reaches the left border.")
        else:
            print("BORDER_STOP_LEFT 0:The motor continues to move when it reaches the left border.")

        if (edgst.BorderFlags & BorderFlags.BORDER_STOP_RIGHT):
            print("BORDER_STOP_RIGHT 1:The motor stops when it reaches the rigth border.")
        else:
            print("BORDER_STOP_RIGHT 0:The motor continues to move when it reaches the rigth border.")

        if (edgst.BorderFlags & BorderFlags.BORDERS_SWAP_MISSET_DETECTION):
            print("BORDERS_SWAP_MISSET_DETECTION 1:Detection of incorrect setting of limit switches is enabled.")
        else:
            print("BORDERS_SWAP_MISSET_DETECTION 0:Detection of incorrect setting of limit switches is disabled.")

        # EnderFlags
        Ender_Flags = ["EnderFlags", "ENDER_SWAP", "ENDER_SW1_ACTIVE_LOW", "ENDER_SW2_ACTIVE_LOW"]
        print("EnderFlags {0:x}".format(edgst.EnderFlags))
        if (edgst.EnderFlags & EnderFlags.ENDER_SWAP):
            print("ENDER_SWAP 1:The first limit switch is located on the right.")
        else:
            print("ENDER_SWAP 0:The first limit switch is located on the left.")

        if (edgst.EnderFlags & EnderFlags.ENDER_SW1_ACTIVE_LOW):
            print("ENDER_SW1_ACTIVE_LOW 1:Low-level SW1 triggering.")
        else:
            print("ENDER_SW1_ACTIVE_LOW 0:High-level SW1 triggering.")

        if (edgst.EnderFlags & EnderFlags.ENDER_SW2_ACTIVE_LOW):
            print("ENDER_SW2_ACTIVE_LOW 1:Low-level SW2 triggering.")
        else:
            print("ENDER_SW2_ACTIVE_LOW 0:High-level SW2 triggering.")

        # The position of the boundaries.
        print("The positions of the borders")
        print("Coordinate of the left border:Pos {0}, uPos {1}".format(edgst.LeftBorder, edgst.uLeftBorder))
        print("Coordinate of the right border:Pos {0}, uPos {1} \n".format(edgst.RightBorder, edgst.uRightBorder))

        # Enter the values for the flags.
        print("Enter the values for the flags 0 or 1.")
        print("Leave the value unchanged any k.")
        edgst.BorderFlags = input_flags(edgst.BorderFlags, Border_Flags)
        edgst.EnderFlags = input_flags(edgst.EnderFlags, Ender_Flags)

        # Enter borders.
        print("To enter the border Y/N ?")
        key_press = getch()
        if ord(key_press) == 89 or ord(key_press) == 121:  # Press "Y"
            print("Enter borders.")
            try:
                edgst.LeftBorder = int(input_new("Enter the left border: "))
                edgst.RightBorder = int(input_new("Enter the right border: "))
            except:
                print("Left border {0}, right border {1}".format(edgst.LeftBorder, edgst.RightBorder))
        result = lib.set_edges_settings(device_id, byref(edgst))


def test_move_settings(lib, device_id, mode=1):
    """
    Setting up movement parameters.

    Follow the on-screen instructions to change the settings.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
        mode: data mode in feedback counts or in user units. (Default value = 1)
    """

    # Create move settings structure
    print("\nGet motion settings")
    if mode:
        mvst = move_settings_t()
    else:
        mvst = move_settings_calb_t()
    test_get_move_settings(lib, device_id, mvst, mode)

    # Input settings
    try:
        if mode:
            speed = int(input_new("Input speed: "))
            asel = int(input_new("Input acceleration: "))
            decel = int(input_new("Input deceleration: "))
        else:
            speed = float(input_new("Input speed: "))
            asel = float(input_new("Input acceleration: "))
            decel = float(input_new("Input deceleration: "))

        # Filling in the structure move_settings_t
        mvst.Speed = speed
        mvst.Accel = asel
        mvst.Decel = decel

        # Writing data to the controller
        test_set_move_settings(lib, device_id, mvst, mode)
        print("")
    except:
        print("Input error. Enter an integer.")


def input_new(s=""):
    """
    Wrapper for the data entry function.

    Args:
        s: Description of the input value(Default value = "")
    """

    return input(s)


def test_user_unit_mode(lib, device_id):
    """
    User unit mode settings

    After setting this multiplier, you can use special commands with the suffix _calb to set the movement in mm or degrees.
    Follow the on-screen instructions to change the settings.

    Args:
        lib: structure for accessing the functionality of the libximc library.
        device_id: device id.
    """

    print("\nUser unit mode settings.")
    print("User unit coordinate multiplier = {0} \n".format(user_unit.A))
    try:
        fl_val = float(input_new("Set new coordinate multiplier = "))
        user_unit.A = fl_val
        # user_unit.MicrostepMode the value is set together with eng.MicrostepMode

    except:
        print("User unit coordinate multiplier = ", user_unit.A)

