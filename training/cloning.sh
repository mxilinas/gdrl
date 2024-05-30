#!/bin/bash

args=(
"--env_path=../export/train.x86_64"
"--demo_files=../godot-gdrl/demos/demo.json"
"--experiment_name="just_cloning""
"--bc_epochs=1000"
"--eval_episode_count=1000"
"--run_eval_after_training"
"--save_model_path./"
"--onnx_export_path=./"
"--viz"
)

python sb3_imitation.py "${args[@]}"
