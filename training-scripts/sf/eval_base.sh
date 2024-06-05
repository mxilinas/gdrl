#!/bin/bash

args=(
"--env_path=../environments/eval_sync.x86_64"
"--experiment_name='eval'"
"--resume_model_path=../models/sync"
"--inference"
"--viz"
"--timesteps=100000"
)

python sb3_base.py "${args[@]}"
