#!/bin/bash
curr_dir=$PWD ## get current working directory

n_trials=3 #3 # number experimental trials to consider
task_soln=-110 # solution threshold for control task (-110 for mountain car)
n_ep=700 # number of simulated episodes to consider
exp_dirs="results/mcar/angc/;/results/mcar/random/" # semicolon delimited list of experimental output dirs
model_names="ANGC;Random" # semicolon delimited list of model names to appear in legend
plot_fname="results/mcar/mcar_returns.png" # /path/to/plot_name.png
trial_data_prefix="rewards" # prefix name used for each trial numpy array
                            # i.e., inside a dir should be something like: rewards0.npy, rewards1.npy, etc.

## run plotting code
smooth_type="window_mar" ## window_mar = Rainbow-RL-style window avg, weight_mar = exp moving avg
python src/plot_reward.py --exp_dir=$exp_dirs --n_trials=$n_trials --plot_mar=$smooth_type \
                          --plot_fname=$plot_fname --trial_fname_tag=$trial_data_prefix \
                          --num_ep=$n_ep --soln_thresh=$task_soln --model_names=$model_names \
                          --y_min=-201 --y_max=-99 --pwd=$curr_dir

exp_dirs="results/mcar/angc/" # "./"
model_names="ANGC"
plot_fname="results/mcar/mcar_epi_returns.png"
y_lab="Episodic Epist. Signals"
trial_data_prefix="epi_rewards"
smooth_type="none"
python src/plot_reward.py --exp_dir=$exp_dirs --n_trials=$n_trials --plot_mar=$smooth_type \
                          --plot_fname=$plot_fname --trial_fname_tag=$trial_data_prefix \
                          --num_ep=$n_ep --model_names=$model_names --y_min=0.0 --y_max=3.0 \
                          --y_axis_label="$y_lab" --pwd=$curr_dir

exp_dirs="results/mcar/angc/"
model_names="ANGC"
plot_fname="results/mcar/mcar_xpos.png"
y_lab="Final Positions (X-Coord)"
trial_data_prefix="x_positions"
smooth_type="none"
python src/plot_reward.py --exp_dir=$exp_dirs --n_trials=$n_trials --plot_mar=$smooth_type \
                          --plot_fname=$plot_fname --trial_fname_tag=$trial_data_prefix \
                          --num_ep=$n_ep --soln_thresh=0.5 --model_names=$model_names \
                          --y_min=-0.65 --y_max=0.65 --y_axis_label="$y_lab" --pwd=$curr_dir
