
# Code for controlling Shawn's stuff

I'll write something here once I have a reason to.
-Andrew


### Plan for Grid:

1. Identify all the controllers based off of their serial numbers
2. Turn this into an interface where I can control all motors simultaneously

I want to be able to call a setup function from the command line,
and it returns an object with an interface for each axis, like `motors.y`

Once I have that, I want to be able to call functions like the test functions
E.g.: motors.n.move_to(position), motors.n.get_status
- I should keep in mind that I could run into problems with precision
  - I could look into user units and that graph things for calibration
- Would it be smart or dumb to make everything asynchronous?

After that, it shouldn't be too hard to set up a loop like:
```
for x in range(...):
    for y in range(...):
        match getch():
            case 'spacebar':
                motors.x.move_to(x)
                motors.y.move_to(y)
            case 'escape': exit()
```