#!/bin/bash

args=(
"--env_path=../environments/sync.x86_64"
"--experiment_name='sync'"
"--save_model_path=../models/sync.onnx"
"--onnx_export_path=../models/sync.onnx"
"--viz"
"--parallel=10"
"--speedup=100"
"--save_checkpoint_frequency=10000"
"--timesteps=100000"
)

python sf.py "${args[@]}"
