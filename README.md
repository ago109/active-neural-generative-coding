# <p align="center">Active Neural Generative Coding (ANGC)</p>
## <p align="center">Biologically-plausible, Backpropagation-free Learning for Active Inference Neural Control Agents</p>
Implementation of the proposed <a href="https://ojs.aaai.org/index.php/AAAI/article/view/19876">active neural generative coding (ANGC) algorithm</a> (AAAI 2022) for training a simple neurobiologically-plausible cognitive control agent. This work instantiates a form of active inference in terms of predictive coding (PC) circuitry, resulting in an agent that learns by dynamically evolving two PC neural circuits -- an actor/policy circuit and a world/transition model -- across a stream of episodes. Note that this agent is meant for tackling simple dense and sparse reward control problems. The instrumental term (or prior preference) in ANGC is treated to be the problem's (sparse) reward function (reflecting an assumption based on the complete class theorem) while the agent's world model circuit produces an epistemic foraging term. This code specifically provides the mountain car experiments conducted to support this work, in the context of Markov decision processes.

<p align="center">
  <img height="250" src="fig/angc_theory.png"><br>
  <i>Theoretical depiction of an ANGC agent.</i>
</p>


Some key features of ANGC include:
1. Unlike classical DQN, ANGC will stay near the goal state and does not empirically suffer from stability issues.
2. ANGC does not require the design of a dense reward function (or reward engineering to replace a sparse reward function, which is a common practice in RL). This means that ANGC agents can work with  sparse rewards, as shown in our repo.

# Requirements
Our implementation is easy to follow and, with knowledge of basic linear algebra, one can decode the inner workings of ANGC. Please consult our [paper](https://ojs.aaai.org/index.php/AAAI/article/view/19876) for general details on mechanics and experiments conducted at the time.
In this framework, we have provided simple modules; thus making it very convenient to extend our framework.

To run the code, you should only need the following basic packages:
1. Jax (version >= 0.4.18)
2. Numpy
3. Matplotlib
4. Python (version >=3.11.4)
5. [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (version >= 0.29.1)

# Execution

To generate mountain car results as per our paper, simply perform the following steps (running the relevant provided Bash scripts) the following provided Bash scripts:
1. `bash src/run_exp.sh` (This will train/adapt the ANGC agent over `E=700` episodes. Note that this will overwrite the
    pre-computed data found in `results/mcar/angc/`; store this data elsewhere if you want to keep it.)
2. `bash src/plot_exp_stats.sh` (This will produce visuals: a reward plot, an epistemic signal plot, and an agent position plot.)

After running the above two scripts, you can find the simulation outputs in the example
experimental results directory tree that has been pre-created for you.
`results/mcar/angc/` contains the numerical array results for the ANGC model (over 3 experimental trials),  while
`results/mcar/` will contain the plots.

In `results/mcar/`, the following is stored:
* `mcar_returns.png` - plot of episodic rewards (with the mountain car solution threshold
   as well as random baseline lower bound)
* `mcar_epi_returns.png` - plot of episodic cumulative epistemic signals produced for ANGC trained on
  the mountain car problem
* `mcar_xpos.png` - plot of ANGC agent car position (x-coordinate; for mountain car only)

In `results/mcar/angc/`, the following arrays are stored:
* `rewards0.npy`, `rewards1.npy`, `rewards2.npy` - episodic reward arrays for trials 0, 1, and 2
* `epi_rewards0.npy`, `epi_rewards1.npy`, `epi_rewards2.npy` - episodic reward arrays for trials 0, 1, and 2
* `x_position0.npy`, `x_position1.npy`, `x_position2.npy` - agent car positions for trials 0, 1, and 2
(Note that the bash scripts detect working directory you have checked out this code-base.)

Model-specific and training-specific hyper-parameter defaults can be set/adjusted in `src/train_agent.py`.
Note that if you just execute `bash src/plot_exp_stats.sh`, you will obtain plots based on the
pre-computed experimental values for trials 0, 1, and 2 directly. If you do so, you should obtain plots
roughly as follows:

With the provided experimental data, you should obtain the following episodic
reward/return plot (Left; smoothened), episodic cumulative epistemic signal plot (Middle; un-smoothened),
and the agent car position plot (Right; un-smoothened):

<p float="left">
  <img src="fig/mcar_returns.png" width="265" />
  <img src="fig/mcar_epi_returns.png" width="265" />
  <img src="fig/mcar_xpos.png" width="265" />
</p>


Tips while using this algorithm/model on other environments/worlds:
1. Tinker with `gamma` (discount factor, values in range `[0.95,0.99]`) as well as
   with `batch_size` and `eta` (learning rate / synaptic adjustment rate).
2. Changing the architecture of the circuits (i.e., adjusting the number of neurons in each layer)
   can help as well. Currently, inside the system, leaky relu is used for the NGC circuits, but this
   can be modified in the code to use other activation functions supported in JAX, so long
   as the derivative of the activation is also coded in.
3. Track your local losses, which will help in the optimization. Play with integrating different optimizers (beyond the Adam and SGD variants provided in this repo) such as AdamW and RMSprop to speed up the convergence.

# Citation

If you use or adapt (portions of) this code/algorithm in any form in your project(s), or
find the ANGC agent algorithm helpful in your own work, please cite this code's source paper:

```bibtex
@inproceedings{ororbia2022backprop,
  title={Backprop-free reinforcement learning with active neural generative coding},
  author={Ororbia, Alexander G and Mali, Ankur},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  pages={29--37},
  year={2022}
}
```
