"""
Runs the latest experiment. Also launches best_individual.py to show the best individual so far
and plots fitness over generations after the run is over. It handles:
- Parameter configuration via command-line arguments
- Single or multiple optimization runs
- Real-time visualization of best individuals
- Fitness progression plotting
- Output organization with timestamped/custom experiment directories

Example usage:
python experiment.py --gens 200 --runs 5 --sigma 0.8 --plot --visualize_best

Dependencies: 
- Custom modules: run_cmaes, best_individual_latest, plot_fitness_over_gens
- Requires full neuroevolution project environment configuration

By Thomas Breimer
March 6th, 2025

Modified by Takumi Kojima
Apr 28th, 2025
"""

import argparse
import multiprocessing
import time
from datetime import datetime

import run_cmaes
import best_individual_latest
import plot_fitness_over_gens
from snn.model_struct import SPIKE_DECAY_DEFAULT


def main():
    parser = argparse.ArgumentParser(
        description="Run a full evolutionary experiment.")

    # Core params
    parser.add_argument(
        '--mode',
        choices=["h", "s", "v", "b"],
        default="h",
        help='Execution mode: h=headless, s=screen, v=video, b=both')
    parser.add_argument('--gens',
                        type=int,
                        default=100,
                        help='Number of generations to run')
    parser.add_argument('--sigma',
                        type=float,
                        default=1.0,
                        help='Sigma value (exploration strength)')
    parser.add_argument('--runs',
                        type=int,
                        default=1,
                        help='Number of independent CMA-ES runs')
    parser.add_argument('--spike_decay',
                        type=float,
                        default=SPIKE_DECAY_DEFAULT,
                        help='Neuron spike decay rate (default=0.01)')
    parser.add_argument(
        '--robot_config',
        type=str,
        help='Path to robot config file (default=bestbot.json)',
        default="bestbot.json")
    parser.add_argument('--hidden_sizes',
                        type=int,
                        nargs='+',
                        default=[2],
                        help='list of hidden layer sizes')
    parser.add_argument(
        '--snn_input',
        type=str,
        default="ground_dist",
        help='What SNN inputs are used. Options: ["corners", "all_dist"]')

    # Output params
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='Optional experiment folder name (otherwise timestamped)')
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Whether to plot fitness over generations after running')
    parser.add_argument(
        '--visualize_best',
        action='store_true',
        help='Whether to visualize best individual live during training')
    parser.add_argument(
        '--logs',
        type=str,
        default="false",
        help='Whether to generate SNN logs during visualization (true/false)')

    args = parser.parse_args()

    # Resolve experiment name
    if args.exp_name:
        exp_folder = args.exp_name
    else:
        exp_folder = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # Decide whether to run one or multiple CMA-ES runs
    if args.runs == 1:
        print("[INFO] Starting single CMA-ES run...")
        run_cmaes.run(args.mode,
                      args.gens,
                      args.sigma,
                      hidden_sizes=args.hidden_sizes,
                      output_folder=exp_folder,
                      run_number=1,
                      spike_decay=args.spike_decay,
                      robot_config_path=args.robot_config,
                      snn_input_method=args.snn_input)
    else:
        print(f"[INFO] Starting {args.runs} CMA-ES runs...")
        for run_number in range(1, args.runs + 1):
            run_cmaes.run(args.mode,
                          args.gens,
                          args.sigma,
                          hidden_sizes=args.hidden_sizes,
                          output_folder=exp_folder,
                          run_number=run_number,
                          spike_decay=args.spike_decay,
                          robot_config_path=args.robot_config,
                          snn_input_method=args.snn_input)

    # visualization
    if args.visualize_best:
        print("[INFO] Launching live visualization of best individual...")

        logs_bool = args.logs.lower() in ('yes', 'true', 't', '1')

        # Spawn visualization in a separate process
        process = multiprocessing.Process(
            target=best_individual_latest.visualize_best,
            args=(args.mode, logs_bool))
        process.start()
        time.sleep(2)  # Give some time to set up visualization

    # plotting
    if args.plot:
        print("[INFO] Plotting fitness over generations...")
        try:
            filename = f"genomes/{exp_folder}/run_1.csv"
            plot_fitness_over_gens.plot(filename)
        except Exception as e:
            print(f"[WARN] Could not plot fitness: {e}")

    print("[INFO] Experiment completed.")


if __name__ == "__main__":
    main()
