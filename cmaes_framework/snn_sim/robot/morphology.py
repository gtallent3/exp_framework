"""
Representation of an evogym robot.

Authors: Thomas Breimer, Matthew Meek
February 21st, 2025
"""

import os
import math
import numpy as np
from evogym import WorldObject
from snn_sim.robot.actuator import Actuator

ROBOT_SPAWN_X = 0
ROBOT_SPAWN_Y = 10
ENV_FILENAME = "simple_environment.json"


class Morphology:
    """
    Our own internal representation of an evogym robot.
    """

    def __init__(self, filename: str):
        """
        Given an evogym robot file, constructs a robot morphology.

        Parameters:
            filename (str): Filename of the robot .json file.
        """

        self.robot_filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "world_data",
            filename)
        self.get_config()
        self.actuators = self.create_actuator_voxels(self.structure)

    def get_config(self):
        """
        Inits robot characteristics.

        Parameters:
            robot_filepath (str): The filename of the robot to get.
        """
        robot = WorldObject.from_json(self.robot_filepath)
        self.structure = robot.get_structure()
        self.connections = robot.get_connections()

    def create_actuator_voxels(self, structure: np.ndarray) -> list:
        """
        Given a robot structure, creates vertices. Also sets the top left
        and bottom right indicies.

        Parameters:
            structure (np.ndarray): array specifing the voxel structure of the object.

        Returns:
            list: A list of actuator objects.
        """

        # Evogym assigns point mass indices by going through the structure array
        # left to right, top to bottom. The first voxel it sees, it assigns its
        # top left point mass to index zero, top right point mass to index one,
        # bottom left point mass to index two, and bottom right point mass to
        # index three. This pattern continues, expect that any point masses that
        # are shared with another voxel and have already been seen are not added to
        # the point mass array. This script goes through this process, constructing
        # the point mass array and identifying shared point masses to create correct
        # actuator objects.

        # To return, will contain Actuator objects
        actuators = []

        # List of tuples (x, y) corresponding to initial point mass positions and index
        # within this list corresponding the their index when calling robot.get_pos()
        self.point_masses = []

        # Dimensions of the robot
        height = len(structure)
        #length = len(structure[0])

        # Will be the coordinates of the top left point mass of ther current voxel.
        top_y = height
        left_x = 0

        # Follows a similar pattern to point masses, top right actuator is zero,
        # and increments going left to right then top to bottom down the grid
        actuator_action_index = 0

        for row in structure:
            for voxel_type in row:
                if not voxel_type == 0:  # Don't add empty voxels

                    right_x = left_x + 1
                    bottom_y = top_y - 1

                    # Check if top left point mass already in point_masses
                    if (left_x, top_y) in self.point_masses:
                        # If so, find index will be the index of where it already is in the array
                        top_left_index = self.point_masses.index(
                            (left_x, top_y))
                    else:
                        # Else, we make a new point mass position
                        top_left_index = len(self.point_masses)
                        self.point_masses.append((left_x, top_y))

                    # Repeat for top right point mass
                    if (right_x, top_y) in self.point_masses:
                        top_right_index = self.point_masses.index(
                            (right_x, top_y))
                    else:
                        top_right_index = len(self.point_masses)
                        self.point_masses.append((right_x, top_y))

                    # And for bottom left point mass
                    if (left_x, bottom_y) in self.point_masses:
                        bottom_left_index = self.point_masses.index(
                            (left_x, bottom_y))
                    else:
                        bottom_left_index = len(self.point_masses)
                        self.point_masses.append((left_x, bottom_y))

                    # And finally bottom right
                    if (right_x, bottom_y) in self.point_masses:
                        bottom_right_index = self.point_masses.index(
                            (right_x, bottom_y))
                    else:
                        bottom_right_index = len(self.point_masses)
                        self.point_masses.append((right_x, bottom_y))

                    # Voxel types 3 and 4 are actuators.
                    # Don't want to add voxel if its not an actuator
                    if voxel_type in [3, 4]:
                        pmis = np.array([
                            top_left_index, top_right_index, bottom_left_index,
                            bottom_right_index
                        ])
                        actuator_obj = Actuator(actuator_action_index,
                                                voxel_type, pmis)
                        actuators.append(actuator_obj)
                        actuator_action_index += 1

                left_x += 1

            top_y -= 1
            left_x = 0

        self.top_left_corner_index = 0
        self.bottom_right_corner_index = len(self.point_masses) - 1

        return actuators

    def get_corner_distances(self, pm_pos: list) -> list:
        """
        Given the list of robot point mass coordinates generated from sim.object_pos_at_time(),
        returns an list of lists where each top level list corresponds to a an actuator voxel,
        the the sublist contains the distance to the [top left corner, bottom right corner].
        
        Parameters:
            pm_pos (list): A list with the first element being a np.ndarray containing all
                           point mass x positions, and second element containig all point mass
                           y positions.
        
        Returns:
            list: A list of tuples of the distances to the top left point mass and bottom right point 
            mass of each actuator.
        """

        actuator_distances = []

        for actuator in self.actuators:
            actuator_distances.append(
                actuator.get_distances_to_corners(
                    pm_pos, self.top_left_corner_index,
                    self.bottom_right_corner_index))

        return actuator_distances

    def get_actuator_distances(self, pm_pos: list) -> list:
        """
        Given the list of robot point mass coordinates generated from sim.object_pos_at_time(),
        returns an list of lists where each top level list corresponds to a an actuator voxel,
        the the sublist contains the distance to all other voxels.
        
        Parameters:
            pm_pos (list): A list with the first element being a np.ndarray containing all
                           point mass x positions, and second element containig all point mass
                           y positions.
        
        Returns:
            list: A list of list of the distances to all other voxels.
        """

        distances = []

        for this_actuator in self.actuators:
            x1, y1 = this_actuator.get_center_of_mass(pm_pos)
            this_actuator_distances = []

            for other_actuator in self.actuators:
                if not this_actuator == other_actuator:
                    x2, y2 = other_actuator.get_center_of_mass(pm_pos)
                    this_actuator_distances.append(
                        math.sqrt((x2 - x1)**2 + (y2 - y1)**2))

            distances.append(this_actuator_distances)

        return distances

    def get_distance_to_ground(self, pm_pos: list) -> list:
        """
        Given the list of robot point mass coordinates generated from sim.object_pos_at_time(),
        returns an list of lists where each top level list corresponds to a an actuator voxel,
        the the sublist contains the distance to the ground.

        Parameters:
            pm_pos (list): A list with the first element being a np.ndarray containing all
                           point mass x positions, and second element containig all point mass
                           y positions.
            
        Returns:
            list: A list of list of the distances from active voxels to the ground.

        """

        actuator_distances_to_ground = []

        for actuator in self.actuators:
            _, y = actuator.get_center_of_mass(pm_pos)  # the distance to the ground is the y-coord
            actuator_distances_to_ground.append((y,))   # making it a tuple so it can easily work with the corner and ground distance
            
        # print (actuator_distances_to_ground)
        return actuator_distances_to_ground


    def get_corner_and_ground_distance(self, pm_pos: list) -> list:   
        """
        Given the list of robot point mass coordinates generated from sim.object_pos_at_time(),
        returns an list of lists where each top level list corresponds to a an actuator voxel,
        the the sublist contains the distance the ground as well as to top left and bottom right corners.

        Parameters:
            pm_pos (list): A list with the first element being a np.ndarray containing all
                           point mass x positions, and second element containig all point mass
                           y positions.
            
        Returns:
            list: A list of list of the distances from active voxels to the top left, bottom right and the ground.

        """

        all_corner_and_ground_disances = []

        corner_distances = self.get_corner_distances(pm_pos)        # get distances to top left and bottom right corners
        ground_distances = self.get_distance_to_ground(pm_pos)      # get distance to the ground

        for corner, ground in zip(corner_distances, ground_distances):
            corner_and_ground_disances = corner + ground            # combine the corner and ground distances
            all_corner_and_ground_disances.append(corner_and_ground_disances)

        return all_corner_and_ground_disances
