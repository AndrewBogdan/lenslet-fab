
# Code for controlling Shawn's stuff

I'll write something here once I have a reason to.
-Andrew


## To-Do List

0. Quick Fixes
   - Make the 'wrong position' error more verbose
     - Also use the libximc wait until complete funciton
   - Stop the toolpath when wifi disconnects
1. Plotting
   - Automatically adjust marker size based on number of elements (or maybe 
       the minimum distance between them?)
   - Ask Shawn if the rotated toolpath is correct
   - Show points that it can and can't lase when it errors.
2. Harness
    - Make sure we don't lase the same spot twice (a check routine)
    - Send an email to Shawn at finish
    - Allow chaining toolpaths together (maybe in the harness?)
      - But you'd have to change the lasing, and that's not cash
    - Track the original instructions, and expose the new ones, the lases, and 
        the positions in a readable way.
3. Bounds
   - Let them provide a message with each bound, which would be used in the
       error if that bound was hit. For example, "you hit the microscope!".
   - These would also be good to put in a config, you'd need to do it all
        at once, and then it would move its full range and ask you if it's
        still in the same spot.
   - Idea: use the rail to set limits?
   - Allow plotting absolute coordinates too (?)
   - Remove old bound code
   - Implement railing behavior.
4. Spherical Path
   - Arguments: pitch, diameter/radius, and angle from zenith (inside of cone)
   - Tessolate the sphere with triangles (icosahedron-based?)
   - Pitch is spherical/geodesic distance
5. Gas
6. Make a break=True/False option to have movement commands wait until they're
   completed to exit.
7. Switch to using divmod for steps, microsteps (bruh)
   - Test _unit_to_step, make sure it respects microsteps being unsigned.
8. Documentation
   - Fix typehints across all files
   - Get Sphinx or something similar
   - Change get_position_step to be an option in get_position, and have it
       give a MotorPosition type or something.
   - Standardize move_to, move_by, etc.
9. Tests
10. Internet Protection
    - Make a function that runs asynchronously until cancelled. Every second, 
        it pings Google with a timeout of 1 second. If it times out, it raises
        an exception and blocks everything.

Note: There's a pretty circular dependence between sal.py and harness.py, I 
    should change it so that there's a geometry information file that harness
    and sal both access, and move SALC's classmethods there