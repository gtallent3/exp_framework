# History

### Framework Improvements
February 10th, 2025 | By Thomas Breimer | PR approved by Thomas Breimer
- Can save simulations as videos
- Can decide to run in headless, video, window modes from terminal
- Can run individuals
- Can save individuals to videos
- Added some plots

Feb 11th | By Abhay Kashyap
- Added script to transform CMA-ES output to SNN weights and biases
- Script reshapes the flat array into an array of shape (NUM_SNN * PARAMS_PER_SNN)
and then returns a list of dictionaries explicitly separating weights and biases of each SNN

### Introducing snn to evogym pipeline
Thurs. Feb 13th | Luodi Wang | SNN output to evogym through .json file
- Saving the output into a json file from a mock (random) input
- Deleted unused files: node.py, layers.py, net.py

### Command Line Args Improvements
February 13th, 2025 | By James Gaskell & Thomas Breimer
- Added proper named command line arguments for all scripts
- Improved implementation with argparse


### John's SNN Code
February 13th, 2025 | By Atharv Tekurkar
- Removed unecessay elements
- Fixed the functions that are giving errors
- Added unit tests for the functions

### Refactoring and Fixes
Feburary 15, 2025 | By Luodi Wang (partnered with Miguel Garduno on pylint)
- `get-output-state()` method in the snn class from a py file
- Within the SNN folder increased pylint score from 5.54/10 to 10/10 by 80.5%!


Feb 16th | Abhay Kashyap
- Made the `snn_parameters` dictionary elements compliant with the way SNN's `set_weights()` method works
- Completed testing unpacking tests with dummy cmaes output and setting them to SNNs

### Refactoring
Feb 17th | Luodi Wang
- Successfully completed robot configuration with its locations of corners from bestbot.json
- Run inputs through the SNN
- and we've now finally outputted actuator control values

Feb 17th | Abhay Kashyap
- Merged all utility functions for SNN (eg. cmaes output => snn w&b; getting snn outputs, etc.) into one class
- Updated SNN code to fix some duty cycle and firelog errors (duty cycle was always 0 as firelog was not getting updated because of threshold)

Feb 23rd | Abhay Kashyap
- Removed files that were not being used and renames some files to more appropriate names
- Refactored `SNNRunner()` into `SNNController()` and the class simpler to use for simulation - removed redundant functions and converted some functions into private functions for abstraction.

Feb 24th | Thomas Breimer
- Completed integration

### SNN Firing Experiments
Feb 25th | By James Gaskell & Matthew Meek
- Added code to isolate a voxel and record the distances between robot corners
- Output corner values for each time step to csv for analysis
- Graphed expansion and contraction behaviors on a single voxel and large robot to determine SNN firing frequency

Feb 27th | Thomas Breimer
- cmaes bug fixes

Mar 4th | Thomas Breimer
- Added bestbot.json to snn-sim
- Chnaged run_cmaes.py to use bestbot.json

Mar 4th | Guy Tallent
- added run_cmaes2, run_sim2, morph2, actuator2. This maps the point massess to each voxel and allows you to get the distance from each of the active voxels to all four corners of the robot

Mar 4th | Jonathan Fischman
- ring buffer implementation

Mar 5th | Abhay Kashyap
- Created a log to track the frequency of target lengths obtained from SNN
- Added notebook showing a simple visualization of this frequency
- Added code to normalize inputs for better SNN computations

Mar 5th | James Gaskell
- Created best_individual_latest.py which runs the best fitness from the most recent experiment or a specified gen number. Can be changed based on snn type
- Implemented RSNN on new branch using CMA-ES taking into account n previous actuator values for each voxel

March 6th | Thomas Breimer
- added run_experiment.py

### Adding Ring Buffer
March 06th | Atharv Tekurkar
- Added ring buffer
- Implemented it for firelog in the snn structure
- Removed duplicate snn files that were in cmaes_integration

Mar 8th | Abhay Kashyap
- Added length method to ring buffer and made `get()` return `n` elements instead of all if parameter passed
- Added logging of the membrane potential (`level`) to visualize the data accumulation and spikes

Mar 20th | Viet-Dung (Darren) Nguyen
- Added implementation of latency tracking with 2 graphs of fitness over generation and velocity over simulation step

### Ozempic TWO
April 3rd | Matthew Meek & Thomas Breimer
- Removed folders with legacy code. `voxel_length_test`, `rmhc_sim`, `cmaes_demo`, `morpho_demo`.
- Ozempic-ed `cmaes_integration`: removed all code non-essential for it to run.
- Renamed `cmaes_integration` to `cmaes_framework`.
- Reverted framework to old version circa week 9 Winter 2025

### Windows
April 3rd | Thomas Breimer
- Changed csv file naming for Windows compat

### SNN Visualizations
- Created feature branch to investigate SNN firing and levels
- Forked repository and switched to SNN branch

April 6th | Hades Panlilio
- Adjusted weights and neruon resets

### Install instructions
April 7th | Matthew Meek
- updated `INSTALL.md` for getting things working on the sudoless lab machines.

### Pull Request Template
April 11th | John Rieffel
- added pull request template

### SNN Refactor
April 9th | Thomas Bremer
- Added docs for all SNN methods
- Fixed neuron duty cycle
- Added logs for duty cycle and proper fire log
- Added option to generate SNN logs with run_individual.py
- Added graphing notebook for SNN logs

Apr 10th | By Takumi Kojima
- bugfix of best_individual_latest regarding the directory path
- change the default setting best_individual_latest

### best_individual changes
April 14th | James Gaskell
- Edited best individual to create logs
- Updated requirements for the correct version of matplotlib
- Fixed symlink for MacOS

### Data file structure
April 16th | James Gaskell
- Edited the data file structure to include symlink for latest data and grouping of videos, logs and genomes
- Edited creation of logs - one log per genome for tidiness

### Devops instructions
April 25th | Takumi
- Added `DevOps.md` which is a manual of how to use screen command, du command, and ssh to lab computers.

### Data file structure continued
April 25th | James Gaskell
- Edited file structure to collect multiple runs under the same experiment name
- Edited plotting to show fitness over time for mutliple runs of the same experiment
- Moved around data and symlinks to keep everything in the data folderg

### Multiple hidden SNN layers
April 28th | Hades Panlilio
- Added parameter hidden_layers to specify amount of hidden layers/neurons per layer
- Added hidden layer functionality to SpikyNet
- Updated files that run the simulation to accept new parameter and multiple layers

### run_experiment Update & parameterize spiky decay and robot config
April 28th | Takumi
- Updated run_experiment with all possible params for now
- parameterized spiky decay
- parameterized robot_confi

### Supporting alternative SNN Input
April 29th | Thomas Breimer
- Added infrastructure for alternative SNN input options
- Added distance to all other actuators SNN input option

### Updating Repository With Experiment Changes -- Exploring Environment-Based Inputs
June 13th | Guy Tallent
- Added methods to morphology - ground distance and corner + ground distance
- Added snn_input_methods for both of the methods above ^
- Changing the default snn input method for experiments to the ground distance
