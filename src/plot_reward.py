import os
import sys, getopt, optparse
#import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

"""
Code for plotting multi-trial episodic rewards (in numpy form) for reinforcement learning and
active inference biological process models.
Note that your rewards should be saved as a set of reward numpy arrays in a
directory like so:

/path/to/file/<prefix>0.py # <prefix> could be something like "rewards"
/path/to/file/<prefix>1.py
/path/to/file/<prefix>2.py
...

@author: Alexander G. Ororbia (2021)

Usage:
$ python plot_reward.py --exp_dir=output/ --n_trials=1 --plot_mar="window_mar"
                        --plot_fname="mcar_returns.png" --trial_fname_tag="rewards"
                        --num_ep=1000 --soln_thresh=-100 --legend_name=ANGC

exp_dir: directory/path to find the reward numpy arrays
n_trials: how many experimental trials were run
plot_mar: how to plot the moving average over rewards
          (Default: window_mar = window moving average over 100 episodes, Rainbow-RL-style)
plot_fname: the desired filename of output plot
trial_fname_tag: the actual name/string to replace <prefix> above with
num_ep: number episodes to consider w/in the arrays (ideally set this to maximum # episodes simulated)
model_key_name: string to use to name your model in the plot's legend (Default: "RL Model")
"""

def calc_mar_seq(y_seq):
    """
        Applies weighted moving average smoothing to y_seq
    """
    alpha = 0.05 #0.1 #0.1 #0.25 # 0.1
    MAR_e = 0.0
    y_mar = []
    for t in range(len(y_seq)):
        y_t = y_seq[t]
        if t > 0:
            MAR_e = alpha * y_t + (1.0 - alpha) * MAR_e # moving average reward (MAR)
        else:
            MAR_e = y_t
        y_mar.append(MAR_e)
    return y_mar

def calc_mar_window(y_seq, win_len=100): #10):
    """
        Applies moving average computed over a window of win_len prior values to y_seq
        (similar to Rainbow paper moving average used for smoothing)
    """
    window = []
    y_mar = []
    for t in range(len(y_seq)):
        y_t = y_seq[t]
        window.append(y_t)
        if len(window) > win_len:
            window.pop(0)
        # compute window average
        mu = 0.0
        for i in range(len(window)):
            w_i = window[i]
            mu += w_i
        mu = mu / (len(window) * 1.0)
        y_mar.append(mu)
    return y_mar

def calc_subwindow_avg(y_seq, start, end):
    """
        Calculate the average of a particular chunk/sub-window within this
        sequence
    """
    y_mu = 0.0
    for i in range(start, (end+1)):
        y_i = y_seq[i]
        y_mu += y_i
    y_mu = y_mu / ( (end - start) )
    return y_mu

y_min = None
y_max = None
x_axis_label = "# Episodes"
y_axis_label = "Episodic Reward"
model_names = "RL Model"
start = 0
n_trials = 1
plot_mar = False
pwd = ""
exp_dirs = ""
plot_fname = "culum_reward.png"
trial_fname_tag = "_ep_reward"
num_ep = 100 #2000 #10000
soln_thresh = None
options, remainder = getopt.getopt(sys.argv[1:], '', ["exp_dirs=","n_trials=","plot_mar=",
                                                      "plot_fname=","trial_fname_tag=",
                                                      "num_ep=", "model_names=", "soln_thresh=",
                                                      "y_min=", "y_max=", "y_axis_label=",
                                                      "x_axis_label=", "pwd="])
# Collect arguments from argv
for opt, arg in options:
    if opt in ("--exp_dirs"):
        exp_dirs = arg.strip()
    elif opt in ("--pwd"):
        pwd = arg.strip()
    elif opt in ("--n_trials"):
        n_trials = int(arg.strip())
    elif opt in ("--plot_mar"):
        plot_mar = arg.strip().lower() #(arg.strip().lower() == 'true')
    elif opt in ("--plot_fname"):
        plot_fname = arg.strip()
    elif opt in ("--trial_fname_tag"):
        trial_fname_tag = arg.strip()
    elif opt in ("--num_ep"):
        num_ep = int(arg.strip())
    elif opt in ("--model_names"):
        model_names = arg.strip()
    elif opt in ("--soln_thresh"):
        soln_thresh = float(arg.strip())
    elif opt in ("--y_min"):
        y_min = float(arg.strip())
    elif opt in ("--y_max"):
        y_max = float(arg.strip())
    elif opt in ("--y_axis_label"):
        y_axis_label = arg.strip()
    elif opt in ("--x_axis_label"):
        x_axis_label = arg.strip()

out_fname = "{0}".format(plot_fname)
exp_dirs = exp_dirs.split(";")
model_names = model_names.split(";")
## TODO: add dynamic colors (currently only 5 models supported)
colors = ["blue", "red", "green", "brown", "purple"]
d_colors = ["cyan", "salmon", "palegreen", "rosybrown", "plum"]

max_val = 0.0
for d in range(len(exp_dirs)):
    y_mu = None
    y_var = None
    out_dir = exp_dirs[d]
    color = colors[d]
    d_color = d_colors[d]

    print(" Extracting trials from ",out_dir)
    y_trials = None
    # store trial-based data
    for t in range(0, n_trials):
        fname = "{0}/{1}{2}{3}.npy".format(pwd,out_dir,trial_fname_tag,t)
        flag = os.path.exists(fname)
        if flag is True:
            data = np.expand_dims(np.load(fname),axis=1)
            if len(data.shape) > 2:
                data = data[:,:,0]
            print("Trial data shape = ",data.shape)
            if data.shape[0] < num_ep:
                pad = np.ones([num_ep - data.shape[0],1])
                pad = pad * calc_subwindow_avg(data, start=(data.shape[0]-1) - 50, end=data.shape[0]-1)
                data = np.concatenate((data,pad),axis=0)
            data = data[0:num_ep,:]
            yfit = data #data[:,1]

            if plot_mar == "weight_mar": # convert to weighted MAR (moving average reward)
                yfit = calc_mar_seq(yfit)
            elif plot_mar == "window_mar": # convert to window MAR (moving average reward)
                yfit = calc_mar_window(yfit)

            if y_trials is not None:
                y_trials = np.concatenate((y_trials, yfit),axis=1)
            else:
                y_trials = yfit
                if n_trials == 1:
                    y_trials = np.asarray(y_trials)
                    if len(y_trials.shape) == 1:
                        y_trials = np.expand_dims(yfit,axis=1)
        else:
            print("ERROR: {} does not exist!".format(fname))
    # compute means
    y_mu = np.sum(y_trials,axis=1,keepdims=True)/(y_trials.shape[1] * 1.0)
    # compute standard deviations
    y_sd = np.ones(y_mu.shape)
    if y_trials.shape[1] > 1:
        diff = y_trials - y_mu
        y_sd = np.sqrt( np.sum(diff * diff,axis=1,keepdims=True)/(y_trials.shape[1] * 1.0 - 1.0) )
    dyfit = y_sd
    max_val = max(max_val, np.amax(y_mu + y_sd))

    xfit = np.arange(start=start, stop=len(yfit), step=1) + 1
    plt.plot(xfit, np.squeeze(y_mu), '-', color=color)
    #if plot_mar is False:
    plt.fill_between(xfit, np.squeeze(y_mu - dyfit), np.squeeze(y_mu + dyfit), color=d_color, alpha=0.2)

if y_min is None:
    y_min = np.amin(y_mu - y_sd)
if y_max is None:
    y_max = np.amax(y_mu + y_sd)

if soln_thresh is not None:
    xfit = np.arange(start=start, stop=num_ep, step=1) + 1
    yfit = np.ones(num_ep) * soln_thresh #-110.0
    plt.plot(xfit, yfit, '-.', color="gray", linewidth=2)

plt.ylim(y_min, y_max)
x_gap = 5 #1
plt.xlim(0, len(xfit)+x_gap) #;
fontSize = 20 #12
plt.xlabel(x_axis_label, fontsize=fontSize)
plt.ylabel(y_axis_label, fontsize=fontSize)
plt.grid()

models = []
for i in range(len(model_names)):
    model_name = model_names[i]
    models.append( mpatches.Patch(color=colors[i], label=model_name) )


plt.legend(handles=models, fontsize=16, ncol=3,borderaxespad=0, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.175))#, prop=fontP)
plt.tight_layout()
plt.savefig("{0}/{1}".format(pwd,out_fname))
plt.clf()
