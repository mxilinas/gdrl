#!/bin/bash

args=(
"--env_path=../export/train.x86_64"
"--demo_files=../godot-gdrl/demos/demo.json"
"--bc_epochs=250"
"--rl_timesteps=1000"
"--eval_episode_count=1000"
"--viz"
)

python sb3_imitation.py "${args[@]}"
