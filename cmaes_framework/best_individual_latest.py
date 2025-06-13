"""
Visualize the best individual so far during a run, pulling from all CSVs in latest_genome.

Author: James Gaskell, Thomas Breimer
Modified: April 16th, 2025
"""

import os
from pathlib import Path
import pandas as pd
import argparse
import pathlib
import time
import numpy as np
from snn_sim.run_simulation import run

ITERS = 1000
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


PARENTDIR = Path(__file__).parent.resolve()
GENOME_START_INDEX = 3
HIDDEN_SIZES = [2]
GENOME_FOLDER = Path(os.path.join(PARENTDIR,"data","latest_genome"))

def wait_for_file(path):
    """
    Wait for a file to be created and contain valid 'best_fitness' data.
    """
    while True:
        try:
            df = pd.read_csv(path)
            if not df.empty and "best_fitness" in df.columns:
                _ = min(df["best_fitness"])
                return df
        except Exception:
            pass

        print(f"Waiting for valid file: {path}")
        time.sleep(5)


def get_best_from_all_csvs():
    """
    Iterate through all CSV files in GENOME_FOLDER and return the row with the best fitness.
    """
    best_row = None
    best_fitness = float('inf')
    best_file = None

    for csv_file in GENOME_FOLDER.glob("*.csv"):
        df = wait_for_file(csv_file)

        try:
            current_best = df.loc[df["best_fitness"] == min(df["best_fitness"])]
            current_fitness = current_best["best_fitness"].values[0]

            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_row = current_best
                best_file = csv_file

        except Exception as e:
            print(f"Skipping file {csv_file} due to error: {e}")

    return best_row, best_file


def visualize_best(mode, logs):
    """
    Continuously run the best individual across all CSVs.
    """
    os.makedirs("data", exist_ok=True)

    while True:

        best_row, source_file = get_best_from_all_csvs()

        if best_row is not None:
            try:
                genome = best_row.values.tolist()[0][GENOME_START_INDEX:]
                generation = int(best_row.values.tolist()[0][0])
                this_dir = pathlib.Path(__file__).parent.resolve()
                vid_path = os.path.join(this_dir, "data", "videos")

                print(f"\n\n\nFitness: {min(best_row['best_fitness'])}")
                print(f"From file: {source_file.name}")

                vid_name = source_file.stem + "_gen_" + str(generation)

                folder_name = source_file.resolve().parent.name
                log_filename = str(os.path.join(PARENTDIR, "data", "logs", folder_name)) + ".csv"

                # Make video directory if we're making a video.
                if mode in ["v", "b"]:
                    os.makedirs(vid_path, exist_ok=True)
                    run(ITERS, genome, mode, HIDDEN_SIZES, vid_name, vid_path, logs, log_filename, snn_input_method="ground_and_corner_dist")
                    quit()
                elif mode in ["s", "h"]:
                    run(ITERS, genome, mode, HIDDEN_SIZES, vid_name, vid_path, logs, log_filename, snn_input_method="ground_and_corner_dist")
                    if logs:
                        quit()
            except Exception as e:
                print("Error during run:", e)
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--mode', #headless, screen, video, both h, s, v, b
        help='mode for output. h-headless , s-screen, v-video, b-both',
        default="s")
    parser.add_argument(
        '--logs', type=str, help='whether to generate SNN logs (true/false)', default="True")


    args = parser.parse_args()

    logs = args.logs
    if logs.lower() in ('yes', 'true', 't', '1'):
        logs = True
    elif logs.lower() in ('no', 'false', 'f', '0'):
        logs = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    visualize_best(args.mode, logs)
