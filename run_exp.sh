#!/bin/bash
curr_dir=$PWD ## get current working directory

array=(42 69 123) ## exerpimental noise seeds
trial_id=0 ## experimental trial identifier
n_episodes=700 ## number episodes to simulate per experiment
results_dir=results/mcar/angc/ ## results output directory to store arrays

## run experiments across seed array above
for i in "${array[@]}"
do
   python src/train_agent.py --seed=$i --trial_id=$trial_id --n_episodes=$n_episodes \
                             --results_dir=$results_dir --pwd=$curr_dir
   trial_id=$((trial_id+1)) # update trial identifier for next experiment
done
