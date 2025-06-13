"""
Given a genome, runs a simulation of a walking robot in evogym, using an SNN controlled robot,
providing a fitness score corresponding to how far the robot walked.

Author: Thomas Breimer, James Gaskell
January 29th, 2025
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import snn.snn_controller as snn_control
from snn.model_struct import SPIKE_DECAY_DEFAULT
from snn_sim.robot.morphology import Morphology

# Simulation constants
ROBOT_SPAWN_X = 2
ROBOT_SPAWN_Y = 0
ACTUATOR_MIN_LEN = 0.6
ACTUATOR_MAX_LEN = 1.6
FPS = 50
MODE = "v"  # "headless", "screen", or "video"

SNN_INPUT_METHOD_DEFAULT = "ground_and_corner_dist"
DEFAULT_SCALE_SNN_INPUTS = True

FITNESS_OFFSET = 100

# Files
ENV_FILENAME = "bigger_platform.json"
ROBOT_FILENAME = "bestbot.json"
THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def create_video(source, output_name, vid_path, fps=FPS):
    """
    Saves a video from a list of frames

    Parameters:
        source (list): List of cv2 frames.
        output_name (string): Filename of output video.
        vid_path (string): Filepath of output video.
        fps (int): Frames per second of video to save.
    """

    Path(vid_path).mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(os.path.join(vid_path, output_name + ".mp4"),
                          cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (source[0].shape[1], source[0].shape[0]))
    for frame in source:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()


def group_list(flat_list: list, n: int) -> list:
    """
    Groups flat_array into a list of list of size n.

    Parameters:
        flat_list (list): List to groups.
        n: (int): Size of sublists.
    
    Returns:
        list: Grouped list.
    """
    return [list(flat_list[i:i + n]) for i in range(0, len(flat_list), n)]


def run(iters,
        genome,
        mode,
        hidden_sizes,
        vid_name=None,
        vid_path=None,
        snn_logs=False,
        log_filename=None,
        spike_decay=SPIKE_DECAY_DEFAULT,
        robot_config=ROBOT_FILENAME,
        snn_input_method=SNN_INPUT_METHOD_DEFAULT,
        scale_snn_inputs=DEFAULT_SCALE_SNN_INPUTS):
    """
    Runs a single simulation of a given genome.

    Parameters:
        iters (int): How many iterations to run.
        genome (ndarray): The genome of the robot.
        mode (string): How to run the simulation. 
                       "h" runs without any video or visual output.
                       "v" outputs the simulation as a video in the "./videos folder.
                       "s" shows the simulation on screen as a window.
                       "b: shows the simulation on a window and saves a video.
        vid_name (string): If mode is "v" or "b", this is the name of the saved video.
        vid_path (string): If mode is "v" or "b", this is the path the video will be saved.
        snn_logs (bool): Whether to produce SNN logs.
        snn_input_method (str): How SNN inputs are computed. 
                          Options are ["corners", "neighbors"]
        scale_inputs (bool): Whether or not to scale SNN inputs.
    Returns:
        float: The fitness of the genome.
    """

    # Create world
    world = EvoWorld.from_json(
        os.path.join(THIS_DIR, 'robot', 'world_data', ENV_FILENAME))

    robot = WorldObject.from_json(
        os.path.join(THIS_DIR, 'robot', 'world_data', robot_config))

    world.add_from_array(name='robot',
                         structure=robot.get_structure(),
                         x=ROBOT_SPAWN_X + 1,
                         y=ROBOT_SPAWN_Y + 1,
                         connections=robot.get_connections())

    # Create simulation
    sim = EvoSim(world)
    sim.reset()

    # Set up viewer
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    video_frames = []

    # Get position of all robot point masses
    init_raw_pm_pos = sim.object_pos_at_time(sim.get_time(), "robot")

    morphology = Morphology(robot_config)

    robot_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'robot', 'world_data', robot_config)

    NUM_ACTUATORS, SNN_INPUT_SHAPE = snn_control.compute_genome_size(
        robot_file_path, snn_input_method, hidden_sizes)

    if snn_input_method == "corners":
        snn_input_size = 2
    elif snn_input_method == "all_dist":
        snn_input_size = NUM_ACTUATORS - 1
    elif snn_input_method == "ground_dist":
        snn_input_size = 1
    elif snn_input_method == "ground_and_corner_dist":
        snn_input_size = 3


    snn_controller = snn_control.SNNController(snn_input_size,
                                               hidden_sizes,
                                               1,
                                               robot_config=robot_file_path,
                                               spike_decay=spike_decay)

    snn_controller.set_snn_weights(genome)

    def scale_inputs(init, cur):
        init = np.asarray(init, dtype=float)
        cur = np.asarray(cur, dtype=float)

        # Compute relative change
        scaled = ((cur - init) / init) * 10 + 1

        # print("Scaled: ", scaled)

        # Clip so that min is -1 and max is 0
        return scaled

    for i in range(iters):
        # Get point mass locations
        raw_pm_pos = sim.object_pos_at_time(sim.get_time(), "robot")

        # Decide what our inputs to the SNN are going to be

        if snn_input_method == "corners":
            inputs = np.array(morphology.get_corner_distances(raw_pm_pos))
        elif snn_input_method == "all_dist":
            inputs = np.array(morphology.get_actuator_distances(raw_pm_pos))
        elif snn_input_method == "ground_dist":
            inputs = np.array(morphology.get_distance_to_ground(raw_pm_pos))
        elif snn_input_method == "ground_and_corner_dist":
            inputs = np.array(morphology.get_corner_and_ground_distance(raw_pm_pos))

        if i == 0:
            init = inputs

        if scale_snn_inputs:
            inputs = scale_inputs(init, inputs)

        # Get action from SNN controller
        action = snn_controller.get_lengths(inputs)

        # Clip actuator target lengths to be between 0.6 and 1.6 to prevent buggy behavior
        action = np.clip(action, ACTUATOR_MIN_LEN, ACTUATOR_MAX_LEN)

        # Set robot action to the action vector. Each actuator corresponds to a vector
        # index and will try to expand/contract to that value
        sim.set_action('robot', action)

        # Execute step
        sim.step()

        if mode == "v":
            video_frames.append(viewer.render(verbose=False, mode="rgb_array"))
        elif mode == "s":
            viewer.render(verbose=True, mode="screen")
        elif mode == "b":
            viewer.render(verbose=True, mode="screen")
            video_frames.append(viewer.render(verbose=False, mode="rgb_array"))

    viewer.close()

    # Get robot point mass position position afer sim has run
    final_raw_pm_pos = sim.object_pos_at_time(sim.get_time(), "robot")

    fitness = np.mean(final_raw_pm_pos[0]) - np.mean(init_raw_pm_pos[0])

    if mode in ["v", "b"]:
        create_video(video_frames, vid_name, vid_path, FPS)

    if snn_logs:
        snn_controller.generate_output_csv(log_filename)

    return FITNESS_OFFSET - fitness  # Turn into a minimization problem
