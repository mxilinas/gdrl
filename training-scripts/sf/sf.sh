#!/bin/bash

args=(
    "--eval"
    "--batched_sampling=true"
    "--num_workers=1"
    "--trainer=sf"
    "--env=gdrl"
    "--env_path=../environments/multi.x86_64"
    "--experiment_name='multi'"
    "--device=cpu"
    "--num_policies=2"
    "--train_for_env_steps=50000"
    "--viz"
    "--speedup=1"
    "--save_every_sec=600"
    "--env_frameskip=10"
)

gdrl "${args[@]}"
