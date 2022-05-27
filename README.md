
# Code for controlling Shawn's stuff

I'll write something here once I have a reason to.
-Andrew


## To-Do List

0. Plotting
   - Automatically adjust marker size based on number of elements (or maybe 
       the minimum distance between them?)
   - Ask Shawn if the rotated toolpath is correct
1. Harness
    - Make it not in the iPython notebook
    - Make sure we don't lase the same spot twice (a check routine)
    - Give a time estimate of how long it'll take
    - Send an email to Shawn at finish
    - Allow chaining toolpaths together (maybe in the harness?)
      - But you'd have to change the lasing, and that's not cash
    - Track the original instructions, and expose the new ones, the lases, and 
        the positions in a readable way.
2. Bounds
   - Real xy=0 (configurable)
   - This also requires bound-defining
     - Let them provide a message with each bound, which would be used in the
         error if that bound was hit. For example, "you hit the microscope!".
   - These would also be good to put in a config, you'd need to do it all
        at once, and then it would move its full range and ask you if it's
        still in the same spot.
   - Idea: use the rail to set limits?
   - Know beforehand if a toolpath is out of bounds
   - Allow plotting absolute coordinates too (?)
   - Remove old bound code
   - Implement optional railing behavior.
3. Spherical Path
   - Arguments: pitch, diameter/radius, and angle from zenith (inside of cone)
   - Tessolate the sphere with triangles (icosahedron-based?)
   - Pitch is spherical/geodesic distance
   - 3D graph it: Shawn likes the graphing!
   - Need a GO_SPH instruction or something
4. Gas
5. Make a break=True/False option to have movement commands wait until they're
   completed to exit.
6. Switch to using divmod for steps, microsteps (bruh)
   - Test _unit_to_step, make sure it respects microsteps being unsigned.
7. Documentation
   - Fix typehints across all files
   - Get Sphinx or something similar
   - Change get_position_step to be an option in get_position, and have it
       give a MotorPosition type or something.
   - Standardize move_to, move_by, etc.
8. Tests