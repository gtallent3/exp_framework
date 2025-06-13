"""
Module for running SNN outputs with proper input/output handling.

Authors: Abhay Kashyap, Atharv Tekurkar
Modified by: Hades Panlilio, Thomas Breimer
"""

from datetime import datetime
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from snn.model_struct import SpikyNet, SPIKE_DECAY_DEFAULT

# Constants for SNN configuration
MIN_LENGTH = 0.6  # Minimum actuator length
MAX_LENGTH = 1.6  # Maximum actuator length
_current_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_current_file))
ROBOT_DATA_PATH = os.path.join(_project_root, "morpho_demo", "world_data",
                               "bestbot.json")


def is_windows():
    """
    Checks if the operating system is Windows.

    Returns:
        bool: True if the OS is Windows, False otherwise.
    """
    return os.name == 'nt' or sys.platform.startswith('win')


class SNNController:
    """Class to handle SNN input/output processing."""

    def __init__(self,
                 inp_size,
                 hidden_sizes,
                 output_size,
                 robot_config=ROBOT_DATA_PATH,
                 spike_decay=SPIKE_DECAY_DEFAULT):
        """
        
        Initializes an SNN Controller for a given robot and SNN hyperparameters.

        Parameters:
            inp_size (int): Number of inputs for each SNN
            hidden_sizes (list): List of numbers of nodes in hidden layers.
            output_size (int): Number of outputs.
            robot_config (str): A robot's .json file.
        """
        self.snns = []
        self.num_snn = 0  # Number of spiking neural networks (actuators)
        self.inp_size = inp_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.spike_decay = spike_decay
        self._load_robot_config(robot_config)

    def _load_robot_config(self, robot_path):
        """
        Load robot configuration from JSON file and initialize SNN.
        
        Args:
            robot_path (str): Path to robot JSON configuration file
            
        """

        robot_data = self._load_robot_file(robot_path)

        # Count actuators (types 3 and 4)
        self.num_snn = sum(1 for t in robot_data["types"] if t in [3, 4])

        # Initialize SNN with proper dimensions
        self.snns = [
            SpikyNet(input_size=self.inp_size,
                     hidden_sizes=self.hidden_sizes,
                     output_size=self.output_size,
                     spike_decay=self.spike_decay) for _ in range(self.num_snn)
        ]

    def _load_robot_file(self, robot_path):
        """
        Loads a robot file.
        
        Parameters:
            robot_path (str): Path to robot JSON configuration file

        Returns:
            Robot data.
        """

        if not os.path.exists(robot_path):
            raise FileNotFoundError(
                f"Robot configuration file not found: {robot_path}")
        with open(robot_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract robot data
        robot_key = list(data["objects"].keys())[0]
        return data["objects"][robot_key]

    def set_snn_weights(self, cmaes_out):
        """
        Retrieve the flat CMA-ES output and 
        reshape it into a structured format for the SNN's `set_weights()`.

        Returns:
            snn_parameters: A dictionary containing the weights and biases for each SNN.
                        - dict with two elements : 'hidden_layer' and 'output_layer'
                            'hidden_layer' - weights and biases for all nodes in the hidden layer
                            'output_layer' - weights and biases for all nodes in the output layer
                        
                            
        Raises:
            ValueError: If the length of the CMA-ES output does not match the expected size.
        """

        flat_vector = np.array(cmaes_out)

        # Compute parameters for each SNN
        params_per_snn = 0
        layer_input_size = self.inp_size

        # Sum hidden layers
        for hidden_size in self.hidden_sizes:
            params_per_snn += (layer_input_size + 1) * hidden_size
            layer_input_size = hidden_size

        # Output layer
        params_per_snn += (layer_input_size + 1) * self.output_size

        if flat_vector.size != (self.num_snn * params_per_snn):
            raise ValueError(
                f"Expected CMA-ES output vector of size "
                f"{self.num_snn * params_per_snn}, got {flat_vector.size}.")

        reshaped = flat_vector.reshape((self.num_snn, params_per_snn))

        snn_parameters = {}

        for snn_idx, params_per_net in enumerate(reshaped):
            layer_weights = []
            start = 0
            layer_input_size = self.inp_size

            # Hidden layers
            for hidden_size in self.hidden_sizes:
                num_params = (layer_input_size + 1) * hidden_size
                layer_weights.append(params_per_net[start:start + num_params])
                start += num_params
                layer_input_size = hidden_size

            # Output layer
            output_params = params_per_net[start:]

            snn_parameters[snn_idx] = {
                'hidden_layers': layer_weights,
                'output_layer': output_params
            }

        for snn_id, params in snn_parameters.items():
            self.snns[snn_id].set_weights(params)

    def _get_output_state(self, inputs):
        """
        Run SNN with distances from each actuator to corners of robot.
        
        Args:
            inputs (list): A list of tuples of the distances to the top left point mass and bottom right point mass
                           for each actuator in the robot.
            
        Returns:
            dict: Contains 'continuous_actions' and 'duty_cycles' for each SNN.
        """

        outputs = {}
        for snn_id, snn in enumerate(self.snns):

            spikes, levels = snn.compute(inputs[snn_id])

            actions = [1.6 if spikes[0] == 1 else 0.6]

            outputs[snn_id] = {
                "target_length": actions,
                "outputs": spikes[0],
                "levels": levels,
            }

        return outputs

    def get_lengths(self, inputs):
        """
        Returns a list of target lengths (action array).

        Args:
            inputs (list): A list of tuples of the distances to the top left point mass and bottom right point mass
                           for each actuator in the robot.

        Returns:
            list: Target length for each actuator, the "action array".
        """

        out = self._get_output_state(inputs)

        lengths = []

        for _, item in out.items():
            lengths.append(item['target_length'])

        return lengths

    def generate_output_csv(self, log_filename):
        """
        Generates an output csv log file for activation level, firelog, and firing frequency for
        each neuron in each SNN.
        """

        # Get logs
        fire_logs = self.get_fire_log()
        level_logs = self.get_levels_log()
        duty_cycle_logs = self.get_duty_cycle_log()

        # Find how many time steps there were
        steps = len(fire_logs[0]['output'][0])

        # Generate SNN_log csv file
        csv_header = ['SNN', "layer", 'neuron', 'log']
        csv_header.extend([f"step{i}" for i in range(steps)])

        df = pd.DataFrame(columns=csv_header)

        # Iterate through logs to generate csv
        for snn_id, snn in fire_logs.items():
            for layer_name, layer in snn.items():
                for neuron_id, fire_log_data in enumerate(layer):
                    # Make levels log row
                    level_log_data = level_logs[snn_id][layer_name][neuron_id]
                    level_log_row = [
                        str(snn_id),
                        str(layer_name),
                        str(neuron_id), "levellog"
                    ]
                    level_log_row.extend(level_log_data)
                    df.loc[len(df)] = level_log_row

                    # Make fire log row
                    fire_log_row = [
                        str(snn_id),
                        str(layer_name),
                        str(neuron_id), "firelog"
                    ]
                    fire_log_row.extend(fire_log_data)
                    df.loc[len(df)] = fire_log_row

                    # Make duty cycle log row
                    duty_cycle_data = duty_cycle_logs[snn_id][layer_name][
                        neuron_id]
                    duty_cycle_row = [
                        str(snn_id),
                        str(layer_name),
                        str(neuron_id), "dutycyclelog"
                    ]
                    duty_cycle_row.extend(duty_cycle_data)
                    df.loc[len(df)] = duty_cycle_row

        # Generate file
        data_folder = Path(
            os.path.join(_project_root, "cmaes_framework", "data", "logs"))
        Path(data_folder).mkdir(parents=True, exist_ok=True)
        csv_path = os.path.join(data_folder, log_filename)

        df.to_csv(csv_path, index=False)

        link = (os.path.join(_project_root, "cmaes_framework", "data",
                             "latest_log.csv"))

        # Set up latest.csv symlink
        if os.path.exists(link):
            os.remove(link)

        if is_windows():
            os.symlink(csv_path, link)
        else:
            try:
                os.unlink(link)
            except FileNotFoundError:
                pass
            os.system("ln -s " + csv_path + " " + link)

    def get_fire_log(self):
        """
        Return a dictionary with the firelog for each node in the hidden and output
        layers of each SNN in the controller.
        
        Returns:
            dict: Dictionary with structure:
                    {snn_id: {'hidden0': [...], 'hidden1': [...], ..., 'output': [...]}}
        """
        return {
            i: {
                **{
                    f'hidden{j}': [node.fire_log for node in layer.nodes]
                    for j, layer in enumerate(snn.hidden_layers)
                },
                'output': [node.fire_log for node in snn.output_layer.nodes]
            }
            for i, snn in enumerate(self.snns)
        }

    def get_levels_log(self):
        """
        Return a dictionary with the membrane potential levels 
        log for each node in the hidden and output
        layers of each SNN in the controller.
        
        Returns:
            dict: Dictionary with structure:
                    {snn_id: {'hidden0': [levels_log_node_1, levels_log_node_2, ...], 
                    'hidden1': [...], ..., 'output': [levels_log_node_1, levels_log_node_2, ...]}}
        """
        return {
            i: {
                **{
                    f'hidden{j}': [node.get_levels_log() for node in layer.nodes]
                    for j, layer in enumerate(snn.hidden_layers)
                },
                'output': [node.get_levels_log() for node in snn.output_layer.nodes]
            }
            for i, snn in enumerate(self.snns)
        }

    def get_duty_cycle_log(self):
        """
        Return a dictionary with the duty cycle
        log for each node in the hidden and output
        layers of each SNN in the controller.
        
        Returns:
            dict: Dictionary with structure:
                    {snn_id: {'hidden0': [duty_cycle_node_1, duty_cycle_node_2, ...], 
                    'hidden1': [...], ..., 'output': [duty_cycle_node_1, duty_cycle_node_2, ...]}}
        """
        return {
            i: {
                **{
                    f'hidden{j}': [node.get_duty_cycle_log() for node in layer.nodes]
                    for j, layer in enumerate(snn.hidden_layers)
                },
                'output': [node.get_duty_cycle_log() for node in snn.output_layer.nodes]
            }
            for i, snn in enumerate(self.snns)
        }
      
def compute_genome_size(robot_path, snn_input_method, hidden_sizes):
    """
    Given a robot body file an SNN input method, and hidden layer sizes, 
    computes how long the genome needs to be and how many actuators there are.

    Parameters:
        robot_path (str): Path to robot JSON configuration file.
        snn_input_method (str): How SNN inputs are computed. 
                        Options are ["corners", "neighbors"]
        hidden_sizes (list): A list of the sizes of all the hidden layers.

    Returns:
        tuple: (num_actuators (int), genome_length (int)).
    """

    if not os.path.exists(robot_path):
        raise FileNotFoundError(
            f"Robot configuration file not found: {robot_path}")
    with open(robot_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract robot data
    robot_key = list(data["objects"].keys())[0]
    robot_data = data["objects"][robot_key]

    # Count actuators (types 3 and 4)
    num_snn = sum(1 for t in robot_data["types"] if t in [3, 4])

    if snn_input_method == "corners":
        inp_size = 2  # Inputs are distances to two corners
    elif snn_input_method == "all_dist":
        inp_size = num_snn - 1  # Inputs are distances to all other actuators
    elif snn_input_method == "ground_dist":
        inp_size = 1  # Inputs are distances from actuators straight to the ground
    elif snn_input_method == "ground_and_corner_dist":
        inp_size = 3 # Inputs are distances from actuators straight to the ground and to two corners

    # Compute parameters for each SNN
    params_per_snn = 0
    layer_input_size = inp_size

    # Sum hidden layers
    for hidden_size in hidden_sizes:
        params_per_snn += (layer_input_size + 1) * hidden_size
        layer_input_size = hidden_size

    # Output layer
    params_per_snn += (layer_input_size + 1)

    genome_length = num_snn * params_per_snn

    return num_snn, genome_length
