"""
This module is an extended example of using the libximc library to control 8SMC SERIES using the Python language.

Warning:
    The keyboard module tracks clicks even if the window is not active.
    To avoid problems, do not change the focus until you finish working with the example.

# Dependences
  -Necessary Python packages:
    netifaces
    getpass
    keyboard
    getch for linux and macos, if the package keyboard is blocked due to lack of root user rights.
    pyximc.py for correct usage of the library libximc you need to add the file  wrapper with the structures of the library to python path.

  -To search for network devices, you must have a file keyfile.sqlite

  -Required libraries for Windows:
    bindy.dll
    libximc.dll
    xiwrapper.dll

  -Required libraries for Linux:
    libbindy.so
    libximc.so
    libxiwrapper.so
"""

from pyximc import *

from scripts.test_extended.dialogs import device_selection_dialog, \
    device_actions_dialog
from scripts.test_extended.config import eng, user_unit
from scripts.test_extended.info_io import test_info, test_serial

if sys.version_info >= (3, 0):
    import urllib.parse


# cur_dir = os.path.abspath(os.path.dirname(__file__))  # Specifies the current directory.
# ximc_dir = os.path.join(cur_dir, "..", "..", "..", "ximc")  # Formation of the directory name with all dependencies.
# ximc_package_dir = os.path.join(ximc_dir, "crossplatform", "wrappers",
#                                 "python")  # Formation of the directory name with python dependencies.
# sys.path.append(ximc_package_dir)  # add pyximc.py wrapper to python path


def main():
    """
    Main function of the example
    
    Main function opens the device search Manager.
    You connect to the selected device, work with it, and disconnect from the device at the end of the program.
    
    note:
        The device_id parameter in function close_device() is a C pointer, unlike most library functions that use this parameter
    
    Starts Manager search for devices and the General Manager work with the device.

    Args:
    """

    print("Library loaded")

    sbuf = create_string_buffer(64)
    lib.ximc_version(sbuf)
    print("Library version: " + sbuf.raw.decode().rstrip("\0"))

    # The choice of dialogue of the working device.
    open_name = device_selection_dialog()

    # Checking the correct device name.
    if not open_name:
        exit(1)

    if type(open_name) is str:
        open_name = open_name.encode()

    # Open selected device
    print("\nOpen device " + repr(open_name))
    device_id = lib.open_device(open_name)

    if device_id <= 0:
        print("Error open device ")
        exit(1)
    else:
        print("Device id: " + repr(device_id))

    # Device info
    test_info(lib, device_id)
    test_serial(lib, device_id)

    result = lib.get_engine_settings(device_id, byref(eng))
    user_unit.MicrostepMode = eng.MicrostepMode

    # Dialog for selecting an action on the device
    device_actions_dialog(lib, device_id)

    print("\nClosing")
    # The device_t device parameter in this function is a C pointer, unlike most library functions that use this parameter
    lib.close_device(byref(cast(device_id, POINTER(c_int))))
    print("Done")


if __name__ == "__main__":
    main()
