from pyximc import engine_settings_t, calibration_t

user_name = "root"
key_esc = "esc"

# Create engine settings structure
eng = engine_settings_t()
# Create user unit settings structure
user_unit = calibration_t()
user_unit.A = 1;

# mode 0 - movement in user units.
# mode 1 - movement in step or encoder unit