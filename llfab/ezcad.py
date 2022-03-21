"""ezcad.py

Python interface for EZCAD. Meant to abstract away the gross-ness.
"""

import logging

import pyautogui


_logger = logging.getLogger(__name__)

# This is a neutral part of the window to bring EZCAD into focus
EZCAD_LOC = (1100, 100)
EYE_LOC = (602, 95)


def ezcad_lase():
    _pag_ezcad_locate()
    _pag_ezcad_mark()


def _pag_ezcad_locate():
    """Make sure that EZCAD is present on the screen"""
    # TODO: Make the filepath a bit less fragile.
    eye_pos = pyautogui.locateCenterOnScreen(r'LensletFab\etc\ezcad2_eye.png')
    if eye_pos is None:
        raise AutoGUIError(
            #'Unable to bring EZCAD into focus.
            'EZCAD is not in focus. Open EZCAD and click the shape you want '
            'lased.'
        )
    elif abs(eye_pos[0] - EYE_LOC[0]) > 30 or abs(eye_pos[1] - EYE_LOC[1]) > 30:
        raise AutoGUIError(
            f'Found the eye, but it ({eye_pos}) isn\'t in the expected '
            f'location ({EYE_LOC}), move EZCAD to the bottom window, '
            f'maximized but not full-screen.'
        )


def _pag_ezcad_mark():
    _logger.info('Pressing F2, EZCAD should Mark.')
    pyautogui.keyUp('shift')  # Shift key-up in case we're in iPython.
    old_pos = pyautogui.position()  # Return to old mouse position later.
    pyautogui.click(*EZCAD_LOC)  # Click EZCAD.
    pyautogui.press('f2')  # Press F2 to mark.
    new_pos = pyautogui.position()  # If mouse has moved, attempt to account
    pyautogui.moveTo(
        x=old_pos[0] + new_pos[0] - EZCAD_LOC[0],
        y=old_pos[1] + new_pos[1] - EZCAD_LOC[1],
    )
    pyautogui.click()


class AutoGUIError(Exception):
    pass
