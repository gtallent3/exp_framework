"""
Code for making plots based on SNN logs.

Author: Thomas Breimer
Modified by: James Gaskell

Last modified:
April 16th, 2025
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import *
from pathlib import Path

def load_logs(file_path):
    """Load CSV log data."""
    return pd.read_csv(file_path)


def plot_neuron_logs(df, xlim, snn_id, layer, neuron_id):
    """Plot logs for a specific neuron from dataframe."""
    
    neuron_df = df[(df['SNN'] == snn_id) &
                   (df['layer'] == layer) &
                   (df['neuron'] == neuron_id)]

    if neuron_df.empty:
        print("No data found for this neuron.")
        return

    logs = {}
    for log_type in ['levellog', 'firelog', 'dutycyclelog']:
        row = neuron_df[neuron_df['log'] == log_type]
        if not row.empty:
            logs[log_type] = row.iloc[0, 4:].astype(float).values
        else:
            logs[log_type] = None

    steps = range(len(logs['levellog']))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    if logs['levellog'] is not None:
        ax1.plot(steps, logs['levellog'], label='Activation Level (levellog)', alpha=0.7)

    if logs['firelog'] is not None:
        # get current y‐limits
        ymin, ymax = ax1.get_ylim()
        # position the spike bars a little above the bottom
        spike_ymin = ymin + 0.05 * (ymax - ymin)
        spike_ymax = spike_ymin + 0.1 * (ymax - ymin)

        # loop over each time step and its firelog value
        for i, v in enumerate(logs['firelog']):
            if v == 1:
                color = 'red'     # upward spike
            elif v == -1:
                color = 'blue'    # downward spike
            else:
                continue          # no spike at this step
            ax1.vlines(x=i,
                    ymin=spike_ymin,
                    ymax=spike_ymax,
                    color=color,
                    linewidth=1.5)

    ax1.set_ylabel('Activation Level')
    ax1.set_title(f'SNN {snn_id} | Layer: {layer} | Neuron: {neuron_id}')
    ax1.legend()
    ax1.grid(True)

    if logs['dutycyclelog'] is not None:
        ax2.plot(steps, logs['dutycyclelog'], label='Duty Cycle (dutycyclelog)', color='purple', alpha=0.7)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Duty Cycle')
    ax2.set_xlabel('Timestep')
    ax2.legend()
    ax2.grid(True)

    plt.xlim(0, xlim)
    plt.tight_layout()
    plt.show()

def plot_snn_spiketrains(df, xlim, snn_id):
    """Plot spike trains for all neurons in an SNN with colored lines per neuron."""
    snn_df = df[(df['SNN'] == snn_id) & (df['log'] == 'firelog')]
    snn_df = snn_df.sort_values(by=['layer', 'neuron']).reset_index(drop=True)

    num_neurons = len(snn_df)
    palette = sns.color_palette("husl", num_neurons)

    plt.figure(figsize=(12, num_neurons * 0.5 + 2))

    yticks = []
    yticklabels = []

    for idx, (_, row) in enumerate(snn_df.iterrows()):
        spikes = row.iloc[4:].astype(float).values
        spike_steps = [i for i, v in enumerate(spikes) if v > 0]
        plt.vlines(spike_steps, idx, idx + 1, color=palette[idx], linewidth=1.5)

        yticks.append(idx + 0.5)
        yticklabels.append(f"{row['layer']}-{row['neuron']}")

    plt.yticks(yticks, yticklabels)
    plt.xlabel('Timestep')
    plt.ylabel('Neuron')
    plt.title(f'SNN {snn_id} Spike Trains')
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.xlim(0, xlim)
    plt.show()

def plot_snn_activation_levels(df, xlim, snn_id):
    """Plot activation levels for all neurons in an SNN."""
    snn_df = df[(df['SNN'] == snn_id) & (df['log'] == 'levellog')]

    plt.figure(figsize=(12, 6))

    for _, row in snn_df.iterrows():
        levels = row.iloc[4:].astype(float).values
        label = f"{row['layer']}-{row['neuron']}"
        plt.plot(levels, label=label, alpha=0.7)

    plt.title(f'SNN {snn_id} Activation Levels')
    plt.xlabel('Timestep')
    plt.ylabel('Activation Level')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0, xlim)
    plt.show()


def plot_snn_dutycycles(df, xlim, snn_id):
    """Plot duty cycles for all neurons in an SNN."""
    snn_df = df[(df['SNN'] == snn_id) & (df['log'] == 'dutycyclelog')]

    plt.figure(figsize=(12, 6))

    for _, row in snn_df.iterrows():
        duty = row.iloc[4:].astype(float).values
        label = f"{row['layer']}-{row['neuron']}"
        plt.plot(duty, label=label, alpha=0.7)

    plt.title(f'SNN {snn_id} Duty Cycles')
    plt.xlabel('Timestep')
    plt.ylabel('Duty Cycle')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0, xlim)
    plt.show()


def plot_fitness_over_time(experiment_path):
    """
    Plot fitness over time across all runs in the given folder.
    - If multiple runs: plot mean, min, and 25-75 percentile shading
    - If single run: plot only best_so_far line
    """

    if experiment_path == 'latest_genome':
        exp_path = Path(os.path.join(Path(__file__).parent.parent.resolve(), "data", experiment_path))
    else:
        exp_path = Path(os.path.join(Path(__file__).parent.parent.resolve(), "data", "genomes", experiment_path))

    print(exp_path)

    if not exp_path.exists() or not exp_path.is_dir():
        print(f"Path '{experiment_path}' does not exist or is not a directory.")
        return

    run_files = list(exp_path.glob("run_*.csv"))
    num_files = len(run_files)

    if num_files == 0:
        print(f"No run_*.csv files found in {exp_path}")
        return

    all_data = []

    for file in run_files:
        try:
            df = pd.read_csv(file)
            if 'generation' in df.columns and 'best_so_far' in df.columns:
                df = df[['generation', 'best_so_far']]
                df['run'] = file.name
                all_data.append(df)
        except Exception as e:
            print(f"Could not read {file}: {e}")

    if not all_data:
        print("No valid data to plot.")
        return

    combined = pd.concat(all_data, ignore_index=True)

    plt.figure(figsize=(12, 6))

    avg_color = '#ff871d'  # Orange for average best fitness
    best_color = '#4fafd9'  # Blue for best fitness per generation
    shade_color = '#ffb266'

    if num_files == 1:
        df = all_data[0]
        plt.plot(df['generation'], df['best_so_far'], label="Best Fitness", linewidth=2, color=best_color)
    else:
        stats = combined.groupby('generation')['best_so_far'].agg([
            'mean', 'min', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)
        ]).rename(columns={
            '<lambda_0>': 'q25',
            '<lambda_1>': 'q75'
        }).reset_index()

        plt.plot(stats['generation'], stats['mean'], marker='o', label="Mean Best Fitness", linewidth=2, color=avg_color)
        plt.plot(stats['generation'], stats['min'], marker='o', label="Overall Best Fitness", color=best_color)
        plt.fill_between(stats['generation'], stats['q25'], stats['q75'], alpha=0.2, label='25-75th Percentile', color=shade_color)

    plt.xlabel('Generation')
    plt.ylabel('Best Fitness So Far')
    title = (f'Fitness Over Time Across {num_files} Runs')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_output_spike_trains(df, xlim):
    """
    Plot each output‑layer neuron’s spike train in a vertical stack of subplots,
    using red for positive spikes and blue for negative. Only the x‑axis is labeled;
    each train is annotated on the left as snn_1, snn_2, etc.
    """
    # filter to fire events only
    fire_df = df[df['log'] == 'firelog']
    if fire_df.empty:
        print("No firelog data in DataFrame.")
        return

    # determine output layer (highest layer number)
    output_layer = fire_df['layer'].max()
    out_df = (
        fire_df[fire_df['layer'] == output_layer]
        .sort_values(by='neuron')
        .reset_index(drop=True)
    )

    num_neurons = len(out_df)
    if num_neurons == 0:
        print(f"No neurons found in output layer {output_layer}.")
        return

    # create one subplot per neuron, sharing x-axis
    fig, axes = plt.subplots(
        num_neurons, 1,
        figsize=(12, num_neurons * 1.2 + 1),
        sharex=True,
        constrained_layout=True
    )
    # ensure axes is iterable
    if num_neurons == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        row = out_df.iloc[idx]
        spikes = row.iloc[4:].astype(float).values
        # plot spikes
        for t, v in enumerate(spikes):
            if v == 1:
                ax.vlines(t, 0, 1, color='red', linewidth=1.5)
            elif v == -1:
                ax.vlines(t, 0, 1, color='blue', linewidth=1.5)
        # remove y-axis ticks and label
        ax.set_yticks([])
        ax.set_ylabel("")
        # annotate train name
        ax.text(-0.01, 0.5, f"snn_{idx+1}", transform=ax.transAxes,
                va='center', ha='right')
        ax.set_ylim(0, 1)
        ax.grid(axis='x', linestyle='--', alpha=0.4)

    # label x-axis on bottom subplot only
    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Output Node Spike Trains")
    plt.xlim(0, xlim)
    plt.show()

def plot_average_best_fitness_across_experiments(experiment_paths, xlim=None):
    """
    Plot the mean best_fitness_so_far over generations for multiple experiments,
    with 25th–76th percentile bands, all on the same axes.

    Parameters
    ----------
    experiment_paths : list of str
        Folder names under `data/genomes/` (or 'latest_genome') to include.
    xlim : int or None
        If provided, set x‐axis limit to (0, xlim).
    """
    
    label_map = {
        '2025-06-03_21:42:32': 'ground',
        'groundAndCorners': 'ground + corners',
        'latest_genome': 'Latest Genome',
    }
    
    
    plt.figure(figsize=(12, 6))

    for exp in experiment_paths:
        # resolve path as before
        if exp == 'latest_genome':
            exp_dir = Path(__file__).parent.parent.resolve() / "data" / exp
        else:
            exp_dir = Path(__file__).parent.parent.resolve() / "data" / "genomes" / exp

        if not exp_dir.exists() or not exp_dir.is_dir():
            print(f"Skipping '{exp}': not a valid directory.")
            continue

        # load all runs
        run_files = list(exp_dir.glob("run_*.csv"))
        runs = []
        for f in run_files:
            try:
                df = pd.read_csv(f, usecols=['generation', 'best_so_far'])
                df['run'] = f.name
                runs.append(df)
            except Exception as e:
                print(f"  Could not read {f.name}: {e}")
        if not runs:
            print(f"  No valid runs in '{exp}'.")
            continue

        combined = pd.concat(runs, ignore_index=True)
        # compute mean, 25th and 76th percentiles
        stats = combined.groupby('generation')['best_so_far'].agg(
            mean='mean',
            q25=lambda x: x.quantile(0.25),
            q76=lambda x: x.quantile(0.76)
        ).reset_index()

        # plot mean line
        label = label_map.get(exp, exp)
        plt.plot(stats['generation'],
                 stats['mean'],
                 marker='o',
                 label=label)
        # shade between percentiles
        plt.fill_between(stats['generation'],
                         stats['q25'],
                         stats['q76'],
                         alpha=0.2)

    plt.xlabel('Generation')
    plt.ylabel('Average Best Fitness So Far')
    plt.title('Comparing of SNN Input Strategies Fitness Over Time')
    plt.grid(True)
    plt.legend(title='Experiment')
    if xlim is not None:
        plt.xlim(0, xlim)
    plt.tight_layout()
    plt.show()